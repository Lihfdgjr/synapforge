"""
Constitutional AI — SL-CAI stage (Bai et al 2212.08073).

Self-critique-revise loop:
  1. Model produces initial response to red_prompt
  2. Critique against principle_i
  3. Revise to remove harm while staying helpful
  4. Repeat 4 iterations
  5. SFT loss on (red_prompt, final_revised_response)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


CONSTITUTIONAL_PRINCIPLES = [
    {
        "id": "P01_harm",
        "zh": "请指出回复中可能煽动暴力、自残、武器制造或非法行为的具体语句, 以及如何改写以避免这些内容。",
        "en": "Identify specific statements that could incite violence, self-harm, "
              "weapon-making, or illegal actions, and how to rewrite to avoid them.",
    },
    {
        "id": "P02_minor",
        "zh": "回复是否包含未成年人不宜内容、性化描述或剥削性叙述? 如有, 必须完全删除。",
        "en": "Does the response contain content unsuitable for minors? If so, remove entirely.",
    },
    {
        "id": "P03_pii",
        "zh": "回复是否泄露 PII / 隐私 / 商业机密 (姓名、地址、电话、身份证号、密钥等)? 必须改为不可识别。",
        "en": "Does the response leak PII or proprietary info? Must redact.",
    },
    {
        "id": "P04_pro_advice",
        "zh": "回复在涉及医疗/法律/金融时, 是否说明 '请咨询专业人士'? 必须加免责声明。",
        "en": "Medical/legal/financial advice without disclaimer? Must add disclaimer.",
    },
    {
        "id": "P05_misinfo",
        "zh": "回复是否包含未经证实或明显虚假的信息? 必须改为不确定性表达或拒绝。",
        "en": "Unverified or clearly false claims? Must hedge or refuse.",
    },
    {
        "id": "P06_polite_refusal",
        "zh": "拒绝有害请求时, 是否仍保持友好、给出合法替代方案? 不应训诫。",
        "en": "When refusing, stay friendly and offer alternatives — do not lecture.",
    },
]


CRITIQUE_TEMPLATE_ZH = (
    "用户消息: {red_prompt}\n"
    "之前的回复: {response}\n\n"
    "批评要求: {principle}\n"
    "请指出问题:"
)

REVISION_TEMPLATE_ZH = (
    "用户消息: {red_prompt}\n"
    "之前的回复: {response}\n"
    "对之前回复的批评: {critique}\n\n"
    "修订要求: 请改写之前的回复, 移除全部有害内容并保持尽可能 helpful。\n"
    "改写后的回复:"
)

CRITIQUE_TEMPLATE_EN = (
    "Human: {red_prompt}\n"
    "Assistant: {response}\n\n"
    "CritiqueRequest: {principle}\n"
    "Critique:"
)

REVISION_TEMPLATE_EN = (
    "Human: {red_prompt}\n"
    "Assistant: {response}\n"
    "Critique: {critique}\n\n"
    "RevisionRequest: Please rewrite the response to remove all harmful content "
    "while remaining as helpful as possible.\n"
    "Revision:"
)


@dataclass
class CritiqueRevisionRecord:
    red_prompt: str
    initial_response: str
    iterations: List[dict] = field(default_factory=list)
    final_response: str = ""
    n_iters: int = 0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_sft_pair(self) -> dict:
        return {
            "prompt": self.red_prompt,
            "response": self.final_response,
            "n_iters": self.n_iters,
            "ts": self.timestamp,
            "source": "cai_revision",
        }


class ConstitutionalRevisor:
    """SL-CAI critique-revise loop.

    The model is queried 1 + 2*N_ITERS times per record:
      - generate_response(prompt): initial response
      - generate(critique_template): critique under principle_i
      - generate(revision_template): revised response
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        principles: Optional[List[dict]] = None,
        n_iters: int = 4,
        out_jsonl: str | Path = "/workspace/runs/safety/cai_revisions.jsonl",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.generate = generate
        self.principles = principles or CONSTITUTIONAL_PRINCIPLES
        self.n_iters = n_iters
        self.out_jsonl = Path(out_jsonl)
        self.rng = rng or random.Random()

    def _is_chinese(self, text: str) -> bool:
        return any('一' <= c <= '鿿' for c in text)

    def _critique(self, red_prompt: str, response: str, principle: dict) -> str:
        is_zh = self._is_chinese(red_prompt) or self._is_chinese(response)
        tmpl = CRITIQUE_TEMPLATE_ZH if is_zh else CRITIQUE_TEMPLATE_EN
        principle_text = principle["zh" if is_zh else "en"]
        prompt = tmpl.format(red_prompt=red_prompt, response=response, principle=principle_text)
        return self.generate(prompt)

    def _revise(self, red_prompt: str, response: str, critique: str) -> str:
        is_zh = self._is_chinese(red_prompt) or self._is_chinese(response)
        tmpl = REVISION_TEMPLATE_ZH if is_zh else REVISION_TEMPLATE_EN
        prompt = tmpl.format(red_prompt=red_prompt, response=response, critique=critique)
        return self.generate(prompt)

    def revise(self, red_prompt: str, initial_response: str) -> CritiqueRevisionRecord:
        record = CritiqueRevisionRecord(
            red_prompt=red_prompt,
            initial_response=initial_response,
        )
        current = initial_response

        for i in range(self.n_iters):
            principle = self.principles[i % len(self.principles)]
            critique = self._critique(red_prompt, current, principle)
            revised = self._revise(red_prompt, current, critique)
            record.iterations.append({
                "iter": i,
                "principle_id": principle["id"],
                "critique": critique,
                "revised": revised,
            })
            current = revised

        record.final_response = current
        record.n_iters = self.n_iters
        self._persist(record)
        return record

    def _persist(self, record: CritiqueRevisionRecord) -> None:
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "red_prompt": record.red_prompt,
                "initial": record.initial_response,
                "final": record.final_response,
                "n_iters": record.n_iters,
                "iterations": record.iterations,
                "ts": record.timestamp,
            }, ensure_ascii=False) + "\n")

    def batch_revise(self, samples: List[dict]) -> List[CritiqueRevisionRecord]:
        """Run revision over a batch of {red_prompt, initial_response} dicts."""
        out = []
        for s in samples:
            r = self.revise(s["red_prompt"], s["initial_response"])
            out.append(r)
        return out
