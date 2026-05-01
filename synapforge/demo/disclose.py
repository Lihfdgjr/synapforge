"""Honest disclosure banner for the recorded-replay fallback.

When ``synapforge-demo chat`` cannot load a live ckpt and falls back to
``chat_recorded.json``, we MUST tell the audience that what they're seeing
is a recorded transcript from a healthy v4.x snapshot, not the live model.
Investor pitches that show recorded outputs as if they were live look
dishonest the moment somebody asks for a different prompt.

See docs/INSURANCE_NATIVE.md Option B for the full strategic rationale and
docs/ANTI_LORA.md for why "play the recorded Synap" beats "use a different
architecture as a live demo".
"""

from __future__ import annotations


# 2026-05-01 v4.1 vintage. Update both fields together if a fresher healthy
# ckpt is recorded into chat_recorded.json. Keep these honest -- if the
# replay snapshot regresses we have to say so, not paper over it.
_REPLAY_CKPT_DATE = "April 28, 2026"
_REPLAY_CKPT_VINTAGE = "v4.1"
_REPLAY_VAL_PPL = 44
_REPLAY_PARAMS = "100M LNN+SNN"


def disclose_replay() -> str:
    """Return the canonical recorded-replay disclosure string.

    Printed at the top of the chat block when ``run_demo`` falls back to the
    recorded transcript. Wired into ``synapforge.demo.chat_demo.run_demo``
    via the recorded-mode branch -- if you change the wording, also update
    the test that pins it (``tests/integration/test_disclose_replay.py``).
    """
    return (
        "*** Honest disclosure: showing recorded "
        f"{_REPLAY_CKPT_VINTAGE} outputs from "
        f"{_REPLAY_CKPT_DATE} ckpt (val ppl {_REPLAY_VAL_PPL}, "
        f"{_REPLAY_PARAMS}). The v5 / Run 3c trainer "
        "is still converging at the time of this demo. Same architecture, "
        "different ckpt vintage. ***"
    )
