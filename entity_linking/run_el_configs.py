"""Run all unfinished EL configs for a given model.

Hardcode MODEL at the top, then run this script. It checks which of the 6
standard configurations are missing from el-results/ and runs them sequentially.

Usage:
    python entity_linking/run_el_configs.py

You can run three instances simultaneously for the three models:
    # Terminal 1
    python entity_linking/run_el_configs.py   # MODEL = "deepseek-v4-flash"
    # Terminal 2 (edit MODEL first)
    python entity_linking/run_el_configs.py   # MODEL = "gemma4:31b"
    # Terminal 3 (edit MODEL first)
    python entity_linking/run_el_configs.py   # MODEL = "kimi-k2.7-code"
"""

import subprocess
import sys
from pathlib import Path

# ── CONFIGURE THIS ────────────────────────────────────────────────────
MODEL = "qwen3.5:397b"
# ───────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
EL_RESULTS = ROOT / "entity_linking" / "el-results"
NER_INPUT = ROOT / "ner" / "ner-output" / "1816_el_gs" / "1816_el_gs.spacy"
EL_SCRIPT = ROOT / "entity_linking" / "el.py"

# All 6 standard configs: (think, temperature)
CONFIGS = [
    ("false", 0.0),
    ("false", 0.5),
    ("false", 1.0),
    ("low", 1.0),
    ("medium", 1.0),
    ("high", 1.0),
]


def _output_path(think: str, temp: float) -> Path:
    """Construct the expected _el.spacy output path for a given config."""
    slug = MODEL.replace(":", "-").replace("/", "-")
    think_part = f"think{think}" if think and think.lower() not in ("false", "") else "thinkfalse"
    stem = f"1816_el_gs__{slug}_t{temp}_{think_part}_el.spacy"
    return EL_RESULTS / stem


def main() -> None:
    if not NER_INPUT.exists():
        print(f"NER input not found: {NER_INPUT}")
        sys.exit(1)

    pending = []
    for think, temp in CONFIGS:
        out = _output_path(think, temp)
        if out.exists():
            print(f"  ✅  {out.name}  — already exists, skipping")
        else:
            print(f"  ⏳  {out.name}  — not found, queued")
            pending.append((think, temp))

    if not pending:
        print("\nAll configs completed for model:", MODEL)
        return

    print(f"\nRunning {len(pending)} config(s) for model: {MODEL}\n")

    for think, temp in pending:
        label = f"think={think}  temp={temp}"
        print(f"{'=' * 60}")
        print(f"  Starting: {label}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable, EL_SCRIPT,
            "--model", MODEL,
            "--temperature", str(temp),
            "--input", NER_INPUT,
            "--think", think,
        ]

        result = subprocess.run(cmd, cwd=ROOT)

        if result.returncode == 0:
            print(f"\n  ✅  Finished: {label}\n")
        else:
            print(f"\n  ❌  FAILED (exit code {result.returncode}): {label}\n")
            sys.exit(result.returncode)

    print(f"\nAll done for model: {MODEL}")


if __name__ == "__main__":
    main()
