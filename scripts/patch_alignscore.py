"""scripts/patch_alignscore.py

Two surgical patches to ``repos/AlignScore/`` so it works with our
modern transformers / pytorch-lightning versions. AlignScore's ``setup.py``
pins ``transformers<4.40`` and ``pytorch-lightning<2`` — we use newer.

Patches:
    1. ``model.py`` — fall back to ``torch.optim.AdamW`` when
       ``transformers.AdamW`` is missing (top-level export removed
       around transformers 4.41+).
    2. ``inference.py`` — use ``BERTAlignModel.load_from_checkpoint``
       as a true classmethod call (pytorch-lightning 2.x rejects
       calling it on an instance).

Idempotent: on re-run each patch checks whether its replacement is
already present and skips it if so. Exit code 0 unless we hit a real
error (file missing, neither original nor patched marker found).

Run via ``make setup`` after ``pip install -e repos/AlignScore``, or
manually: ``python3 scripts/patch_alignscore.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALIGNSCORE = ROOT / "repos" / "AlignScore" / "src" / "alignscore"


# Each patch is a (path, original, replacement) triple. `original` is the
# exact string we replace; `replacement` is what goes in. The script also
# treats the presence of the replacement string as "already patched".
PATCHES: list[tuple[Path, str, str]] = [
    (
        ALIGNSCORE / "model.py",
        # Original — `AdamW` imported from top-level transformers, removed
        # in 4.41+.
        "from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig",
        # Patched — pull AdamW straight from torch.optim (transformers'
        # version was a thin wrapper over it anyway).
        (
            "from torch.optim import AdamW\n"
            "from transformers import get_linear_schedule_with_warmup, AutoConfig"
        ),
    ),
    (
        ALIGNSCORE / "inference.py",
        # Original — instance-method call, broken under pytorch-lightning 2.x
        (
            "self.model = BERTAlignModel(model=model)"
            ".load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)"
            ".to(self.device)"
        ),
        # Patched — proper classmethod call. ``strict=False`` first to keep
        # diff minimal, ``model=model`` last (forwarded to ``__init__``).
        (
            "self.model = BERTAlignModel.load_from_checkpoint("
            "checkpoint_path=ckpt_path, strict=False, model=model"
            ").to(self.device)"
        ),
    ),
]


def apply(path: Path, original: str, replacement: str) -> str:
    """Apply one patch. Returns a status string for logging."""
    if not path.is_file():
        return f"SKIP {path.name}: file not found ({path})"
    text = path.read_text(encoding="utf-8")
    if replacement in text:
        return f"OK   {path.name}: already patched"
    if original not in text:
        return (
            f"FAIL {path.name}: neither original nor patched marker present — "
            f"upstream may have changed; review manually."
        )
    path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return f"OK   {path.name}: patched"


def main() -> int:
    if not ALIGNSCORE.is_dir():
        print(
            f"AlignScore source dir missing: {ALIGNSCORE}\n"
            f"Did you `git clone https://github.com/yzha/AlignScore "
            f"repos/AlignScore` ?",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for path, original, replacement in PATCHES:
        msg = apply(path, original, replacement)
        print(msg)
        if msg.startswith("FAIL"):
            failures += 1
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
