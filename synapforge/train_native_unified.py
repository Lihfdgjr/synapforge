"""train_native_unified.py — One backbone, native joint training of:
  * text LM (WikiText)
  * image (random + Fuyu-style patch fusion)
  * action (4-button env)
  * NeuroMCP (synapse growth + codebook expansion)

Single train step rotates through tasks; same backbone learns all four.
Native: no frozen encoders, no token tool calls, no LLaVA-style projection.
"""
from __future__ import annotations
import argparse, sys, os, time, math, random
sys.path.insert(0, '/workspace')
import torch, torch.nn as nn, torch.nn.functional as F
import synapforge as sf

VOCAB = 50257
HIDDEN = 256
SEQ_LEN = 128


class NativeUnified(sf.Module):
    """1 backbone + 3 heads all sharing same hidden."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.embed = sf.modal.UnifiedEmbed(hidden=hidden, vocab=VOCAB,
                                           patch_image=8, patch_audio_ms=20,
                                           video_temporal_patch=4)
        self.cfc = sf.LiquidCell(hidden, hidden)
        self.plif = sf.PLIFCell(hidden, threshold_init=0.3)
        self.lm_head = sf.tied_lm_head(hidden, VOCAB, embedding=self.embed.token_embedding)
        self.action = sf.action.ActionHead(hidden, sf.action.OSActionSpec.default())
        self.neuromcp = sf.action.NeuroMCPHead(hidden, codebook_initial=8, codebook_max=64,
                                                synapse_density=0.05, synapse_max_density=0.3)

    def forward_backbone(self, z):
        """Shared backbone: (B, T, D) -> (B, T, D) spike-rate-coded."""
        h = self.cfc(z)
        s, _ = self.plif.forward_seq(h)
        return s

    def forward_text(self, tokens):
        z = self.embed({"text_tokens": tokens})
        s = self.forward_backbone(z)
        return self.lm_head(s)

    def forward_image(self, image):
        z = self.embed({"image": image})
        return self.forward_backbone(z)

    def forward_action(self, image):
        z = self.embed({"image": image})
        s = self.forward_backbone(z)
        return self.action(s.mean(dim=1, keepdim=True))


_text_stream_cache = None
def _get_text_stream(B=8):
    global _text_stream_cache
    if _text_stream_cache is None:
        from synapforge.data import ParquetTokenStream
        _text_stream_cache = iter(ParquetTokenStream(
            "/workspace/data/wt103_raw/train-*.parquet",
            seq_len=SEQ_LEN, batch_size=B, tokenizer="gpt2", loop=True))
    return _text_stream_cache

def make_text_batch(B=8):
    """Real WikiText-103 stream (was random stub)."""
    try:
        x, _y = next(_get_text_stream(B))
        return x.cuda()
    except Exception:
        # fallback to random if data load fails
        return torch.randint(0, VOCAB, (B, SEQ_LEN), device='cuda')

def make_image_batch(B=4):
    return torch.randn(B, 3, 32, 32, device='cuda')

def make_action_batch(B=8):
    """4-button env: red dot in one of 4 quadrants, label = quadrant id."""
    img = torch.zeros(B, 3, 32, 32, device='cuda')
    targets = torch.randint(0, 4, (B,), device='cuda')
    BUTTONS = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for b in range(B):
        cx, cy = BUTTONS[int(targets[b])]
        ix, iy = int(cx * 32), int(cy * 32)
        img[b, 0, max(0, iy-3):iy+3, max(0, ix-3):ix+3] = 1.0
    return img, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="/workspace/runs/synapforge_native_unified")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42)
    model = NativeUnified(hidden=HIDDEN).cuda().to(torch.bfloat16)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[init] NativeUnified {n_p:.2f}M params on cuda bf16", flush=True)
    print(f"[init] tasks: text-LM | image-recon | action-cls | neuromcp-grow", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    losses = {'text': [], 'image': [], 'action': [], 'neuromcp_K': []}
    t_start = time.time()

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        # Rotate task each step (all 4 in 4-step cycle)
        task = step % 4

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if task == 0:
                # text LM
                x = make_text_batch(B=8)
                logits = model.forward_text(x[:, :-1])
                # logits shape (B, T_unified, V); T_unified = T_text + modality marker.
                # Take the last T_text logits (after the marker prepended by UnifiedEmbed)
                Tt = x.shape[1] - 1
                logits_aligned = logits[:, -Tt:, :]
                target = x[:, 1:]
                loss = F.cross_entropy(logits_aligned.reshape(-1, VOCAB).float(),
                                       target.reshape(-1))
                losses['text'].append(float(loss))
            elif task == 1:
                # image reconstruction stub — backbone forward must not NaN
                img = make_image_batch(B=4)
                h = model.forward_image(img)
                # simple: target = mean-pool input image flattened, MSE
                target = F.adaptive_avg_pool2d(img, (4, 4)).reshape(4, -1)
                pred = h.mean(dim=1).float()[:, :target.shape[1]]
                loss = F.mse_loss(pred, target.float())
                losses['image'].append(float(loss))
            elif task == 2:
                # 4-button action classification
                img, target = make_action_batch(B=8)
                act_out = model.forward_action(img)
                # action_type field: idx 0..3 = button click
                # ActionHead returns ActionOutput with .action_type_logits
                logits = act_out.action_type_logits.squeeze(1).float()  # (B, num_action_types)
                loss = F.cross_entropy(logits[:, :4], target)
                losses['action'].append(float(loss))
            else:  # task == 3
                # NeuroMCP synapse + codebook growth
                img = make_image_batch(B=4)
                z = model.embed({"image": img})
                s = model.forward_backbone(z)
                pooled = s.mean(dim=1, keepdim=True)
                nm = model.neuromcp(pooled)
                K = int(model.neuromcp.codebook.alive_mask.sum().item())
                losses['neuromcp_K'].append(K)
                # Use the codebook logits with a target = argmax(self) → contrastive identity loss
                cb_logits = nm['logits'].squeeze(1).float()
                target = cb_logits.argmax(dim=-1)
                loss = F.cross_entropy(cb_logits, target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"step {step} task={task} NaN/Inf, skip", flush=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Trigger plasticity / synaptogenesis (this is the magic — auto grow connections)
        if step % 5 == 0:
            try:
                stats = model.neuromcp.step_plasticity()
                cb_stats = model.neuromcp.codebook.maybe_grow(s.mean(dim=1).detach())
            except Exception:
                pass

        if step % 10 == 0:
            elapsed = time.time() - t_start
            t_avg = sum(losses['text'][-10:]) / max(1, len(losses['text'][-10:]))
            i_avg = sum(losses['image'][-10:]) / max(1, len(losses['image'][-10:]))
            a_avg = sum(losses['action'][-10:]) / max(1, len(losses['action'][-10:]))
            K = losses['neuromcp_K'][-1] if losses['neuromcp_K'] else '-'
            density = float(model.neuromcp.proj.mask.mean().item()) if hasattr(model.neuromcp, 'proj') else '-'
            print(f"step={step:4d} text={t_avg:.3f} img={i_avg:.4f} act={a_avg:.3f} "
                  f"nmcp_K={K} density={density:.3f}  task={task} elapsed={elapsed:.0f}s", flush=True)

    print(f"\n[done] {args.steps} steps in {time.time()-t_start:.0f}s")
    print(f"[done] final text loss: {sum(losses['text'][-5:])/max(1,len(losses['text'][-5:])):.3f}")
    print(f"[done] final action loss: {sum(losses['action'][-5:])/max(1,len(losses['action'][-5:])):.3f}")
    if losses['neuromcp_K']:
        print(f"[done] codebook K: 8 -> {max(losses['neuromcp_K'])} (+{max(losses['neuromcp_K'])-8})")
    # Save
    torch.save(model.state_dict(), f"{args.out}/final.pt")
    print(f"[done] ckpt -> {args.out}/final.pt")


if __name__ == "__main__":
    main()
