"""T2: character-level encoder-decoder for Bengali word -> Banglish spelling (D7-amended).

Trained from random initialization (no pretrained transliteration model):
  1. pretrain on Aksharantar ben_train (1.13M scholarly-convention word pairs) for coverage,
  2. fine-tune on Vashantor train+val word pairs so the output follows the natural
     Banglish convention ("valo", not "bhalo").
Every word type that occurs in the Vashantor TEST split is removed from both training
sets (`model.exclude_test_words`), so Stage 0 measures T2 on words it never saw.
A data tool only: it is not part of the released encoder.

GPU temperature is logged every `monitoring.gpu_log_every_s` (configs/base.yaml, C1);
if it stays above `temp_limit_c` for `temp_sustain_s`, training stops and reports.
Code never touches clocks, power or voltage.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.translit_model --train
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.translit_model          # demo decode
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.common import load_config, repo_path, sha256_file
from data.translit import build_lexicon, load_translit_config, test_word_types
from data.translit_rules import normalize_bengali_word

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
CKPT_NAME = "t2_char.pt"


class CharVocab:
    """Character vocabulary with the 4 special symbols first."""

    def __init__(self, chars: list[str]) -> None:
        self.itos = SPECIALS + sorted(set(chars) - set(SPECIALS))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_words(cls, words: list[str]) -> CharVocab:
        """Build from every character in `words`."""
        return cls(sorted({ch for w in words for ch in w}))

    def encode(self, word: str, max_len: int, bos: bool = False) -> list[int]:
        """Ids for `word`, truncated, ending with EOS (and starting with BOS if asked)."""
        ids = [self.stoi.get(ch, UNK) for ch in word][: max_len - 1 - int(bos)]
        return ([BOS] if bos else []) + ids + [EOS]

    def decode(self, ids: list[int]) -> str:
        """Characters up to the first EOS, specials removed."""
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)

    def __len__(self) -> int:
        return len(self.itos)


class CharSeq2Seq(nn.Module):
    """Transformer encoder-decoder over characters with learned positions."""

    def __init__(self, src_vocab: int, tgt_vocab: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        d = int(cfg["d_model"])
        max_len = max(int(cfg["max_src_chars"]), int(cfg["max_tgt_chars"])) + 2
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.core = nn.Transformer(
            d_model=d, nhead=int(cfg["n_heads"]), num_encoder_layers=int(cfg["enc_layers"]),
            num_decoder_layers=int(cfg["dec_layers"]), dim_feedforward=int(cfg["ffn"]),
            dropout=float(cfg["dropout"]), batch_first=True, norm_first=True,
        )
        self.core.encoder.enable_nested_tensor = False  # irrelevant with norm_first; silences a warning
        self.out = nn.Linear(d, tgt_vocab)
        self.scale = math.sqrt(d)

    def _embed(self, emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return emb(ids) * self.scale + self.pos(pos)[None]

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder memory and its padding mask."""
        mask = src == PAD
        return self.core.encoder(self._embed(self.src_emb, src), src_key_padding_mask=mask), mask

    def decode_step(self, mem: torch.Tensor, mem_mask: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Logits for every target position given the prefix `tgt`."""
        n = tgt.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt.device), 1)
        h = self.core.decoder(self._embed(self.tgt_emb, tgt), mem, tgt_mask=causal, tgt_is_causal=True,
                              tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits."""
        mem, mask = self.encode(src)
        return self.decode_step(mem, mask, tgt_in)


def _pad(seqs: list[list[int]]) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs], dtype=torch.long)


# ----------------------------------------------------------------------------- data

def aksharantar_pairs(cfg: dict[str, Any], exclude: set[str]) -> list[tuple[str, str]]:
    """(bengali, roman) from Aksharantar ben_train; lowercase a-z targets only."""
    out = []
    with zipfile.ZipFile(repo_path(cfg["paths"]["aksharantar_zip"])) as z, z.open("ben_train.json") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            d = json.loads(line)
            b, r = normalize_bengali_word(d["native word"]), d["english word"].strip().lower()
            if b and r.isascii() and r.isalpha() and b not in exclude:
                out.append((b, r))
    return out


def vashantor_pairs(cfg: dict[str, Any], exclude: set[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(train, dev) Vashantor word pairs. Each attested spelling appears min(count, 5)
    times; dev = a fixed fraction of word types (majority spelling), for checkpoint choice."""
    lex = build_lexicon(cfg)
    rng = random.Random(int(cfg["model"]["seed"]))
    words = sorted(w for w in lex if w not in exclude)
    rng.shuffle(words)
    n_dev = int(len(words) * float(cfg["model"]["finetune"]["dev_fraction"]))
    dev_words = set(words[:n_dev])
    train, dev = [], []
    for w in words:
        if w in dev_words:
            dev.append((w, lex[w].most_common(1)[0][0]))
            continue
        for r, c in lex[w].items():
            if r.isalpha():
                train.extend([(w, r)] * min(c, 5))
    return train, dev


# ------------------------------------------------------------------------- training

class GpuThermal:
    """Logs GPU temperature every N seconds; flags a sustained over-limit (C1)."""

    def __init__(self, base: dict[str, Any], log: list[dict[str, Any]]) -> None:
        mon = base["monitoring"]
        self.every, self.limit, self.sustain = float(mon["gpu_log_every_s"]), float(mon["temp_limit_c"]), float(mon["temp_sustain_s"])
        self.log, self.last = log, 0.0
        self.over_since: float | None = None

    def check(self) -> bool:
        """Poll if due; return True if the run must stop (over limit too long)."""
        now = time.time()
        if now - self.last < self.every:
            return False
        self.last = now
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().split(",")
            temp, clock = float(out[0]), float(out[1])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return False
        self.log.append({"t": round(now), "temp_c": temp, "sm_mhz": clock})
        if temp > self.limit:
            self.over_since = self.over_since or now
            return now - self.over_since > self.sustain
        self.over_since = None
        return False


def _lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def _run_phase(model: CharSeq2Seq, pairs: list[tuple[str, str]], src_v: CharVocab, tgt_v: CharVocab,
               mcfg: dict[str, Any], pcfg: dict[str, Any], dev: list[tuple[str, str]] | None,
               thermal: GpuThermal, device: str, tag: str) -> dict[str, Any]:
    rng = random.Random(int(mcfg["seed"]))
    bs, epochs = int(pcfg["batch_size"]), int(pcfg["epochs"])
    total = epochs * math.ceil(len(pairs) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(pcfg["lr"]), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    best: dict[str, Any] = {"dev_acc": -1.0, "state": None, "epoch": 0}
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        # bucket by length for speed: sort within chunks of 100 batches
        chunk = bs * 100
        order = [i for c in range(0, len(order), chunk) for i in sorted(order[c:c + chunk], key=lambda j: len(pairs[j][0]))]
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(batches)
        tot_loss = 0.0
        for b in batches:
            src = _pad([src_v.encode(pairs[i][0], int(mcfg["max_src_chars"])) for i in b]).to(device)
            tgt = _pad([tgt_v.encode(pairs[i][1], int(mcfg["max_tgt_chars"]) + 1, bos=True) for i in b]).to(device)
            for g in opt.param_groups:
                g["lr"] = _lr(step, total, int(pcfg["warmup_steps"]), float(pcfg["lr"]))
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(mcfg["grad_clip"]))
            opt.step()
            tot_loss += loss.item()
            step += 1
            if step % int(mcfg["log_every_steps"]) == 0:
                print(f"  [{tag}] ep {ep} step {step}/{total} loss {loss.item():.3f} {time.time() - t0:,.0f}s", flush=True)
            if thermal.check():
                raise RuntimeError(f"GPU over {thermal.limit} C for > {thermal.sustain} s: stopping (C1); report to Akib")
        msg = f"  [{tag}] epoch {ep} mean loss {tot_loss / len(batches):.4f}"
        if dev:
            acc = sum(h == r for h, r in zip(greedy_decode(model, [w for w, _ in dev], src_v, tgt_v, mcfg, device), [r for _, r in dev])) / len(dev)
            msg += f" dev word acc {acc:.4f}"
            if acc > best["dev_acc"]:
                best = {"dev_acc": acc, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "epoch": ep}
        print(msg, flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"steps": step, "seconds": round(time.time() - t0, 1), "best_dev_acc": best["dev_acc"], "best_epoch": best["epoch"]}


@torch.no_grad()
def greedy_decode(model: CharSeq2Seq, words: list[str], src_v: CharVocab, tgt_v: CharVocab,
                  mcfg: dict[str, Any], device: str, batch_size: int = 1024) -> list[str]:
    """Deterministic batched greedy decoding."""
    model.eval()
    out: list[str] = [""] * len(words)
    order = sorted(range(len(words)), key=lambda i: len(words[i]))
    max_t = int(mcfg["max_tgt_chars"]) + 1
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        src = _pad([src_v.encode(words[i], int(mcfg["max_src_chars"])) for i in idx]).to(device)
        mem, mask = model.encode(src)
        ys = torch.full((len(idx), 1), BOS, dtype=torch.long, device=device)
        done = torch.zeros(len(idx), dtype=torch.bool, device=device)
        for _ in range(max_t):
            nxt = model.decode_step(mem, mask, ys)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if bool(done.all()):
                break
        for j, i in enumerate(idx):
            out[i] = tgt_v.decode(ys[j, 1:].tolist())
    return out


class T2:
    """Loaded T2 model as a fallback callable: list of Bengali words -> spellings."""

    def __init__(self, path: Path | None = None, device: str | None = None) -> None:
        cfg = load_translit_config()
        path = path or repo_path(cfg["paths"]["model_dir"]) / CKPT_NAME
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mcfg = ck["model_cfg"]
        self.src_v, self.tgt_v = CharVocab(ck["src_chars"]), CharVocab(ck["tgt_chars"])
        self.model = CharSeq2Seq(len(self.src_v), len(self.tgt_v), self.mcfg).to(self.device)
        self.model.load_state_dict(ck["state"])
        self.sha256 = sha256_file(path)

    def __call__(self, words: list[str]) -> list[str]:
        return greedy_decode(self.model, words, self.src_v, self.tgt_v, self.mcfg, self.device)


def train(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pretrain on Aksharantar, fine-tune on Vashantor, save the best-dev checkpoint."""
    cfg = cfg or load_translit_config()
    base = load_config(repo_path("configs/base.yaml"))
    mcfg = cfg["model"]
    torch.manual_seed(int(mcfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exclude = test_word_types(cfg) if mcfg["exclude_test_words"] else set()
    ak = aksharantar_pairs(cfg, exclude)
    va_train, va_dev = vashantor_pairs(cfg, exclude)
    src_v = CharVocab.from_words([w for w, _ in ak + va_train + va_dev])
    tgt_v = CharVocab.from_words([r for _, r in ak + va_train + va_dev])
    model = CharSeq2Seq(len(src_v), len(tgt_v), mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"T2: {n_params / 1e6:.2f}M params; aksharantar {len(ak):,} pairs; vashantor train {len(va_train):,} "
          f"/ dev {len(va_dev):,} types; {len(exclude):,} test word types excluded; device {device}", flush=True)
    temps: list[dict[str, Any]] = []
    thermal = GpuThermal(base, temps)
    stats: dict[str, Any] = {"params": n_params, "aksharantar_pairs": len(ak), "vashantor_train_pairs": len(va_train),
             "vashantor_dev_types": len(va_dev), "excluded_test_types": len(exclude)}
    out_dir = repo_path(cfg["paths"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        stats["pretrain"] = _run_phase(model, ak, src_v, tgt_v, mcfg, mcfg["pretrain"], va_dev, thermal, device, "pretrain")
        stats["finetune"] = _run_phase(model, va_train, src_v, tgt_v, mcfg, mcfg["finetune"], va_dev, thermal, device, "finetune")
    except RuntimeError as exc:  # thermal stop (C1): keep the evidence, then re-raise
        stats.update({"aborted": str(exc), "gpu_temp_log": temps})
        (out_dir / "t2_train_aborted.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        raise
    path = out_dir / CKPT_NAME
    torch.save({"state": model.state_dict(), "model_cfg": mcfg, "src_chars": src_v.itos[len(SPECIALS):],
                "tgt_chars": tgt_v.itos[len(SPECIALS):]}, path)
    stats["checkpoint_sha256"] = sha256_file(path)
    stats["gpu_temp_max_c"] = max((t["temp_c"] for t in temps), default=None)
    stats["gpu_temp_log"] = temps
    (out_dir / "t2_train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved {path} sha256 {stats['checkpoint_sha256'][:16]}… max GPU temp {stats['gpu_temp_max_c']} C", flush=True)
    return stats


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.train:
        train()
    else:
        ckpt = repo_path(load_translit_config()["paths"]["model_dir"]) / CKPT_NAME
        if not ckpt.is_file():
            v = CharVocab.from_words(["আমি", "ami"])
            m = CharSeq2Seq(len(v), len(v), load_translit_config()["model"])
            x = _pad([v.encode("আমি", 32)])
            print("untrained demo logits shape:", tuple(m(x, _pad([v.encode("ami", 41, bos=True)])[:, :-1]).shape))
        else:
            t2 = T2()
            words = ["স্বাধীনতা", "বিশ্ববিদ্যালয়", "প্রধানমন্ত্রী", "ভালোবাসা", "কম্পিউটার"]
            print(dict(zip(words, t2(words))))
