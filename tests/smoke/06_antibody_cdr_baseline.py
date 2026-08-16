#!/usr/bin/env python3
"""Phase A smoke test: demo scFv fixture + CDR masks (CPU); optional GPU via generate_antibody.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SMOKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SMOKE_DIR.parents[1]
sys.path.insert(0, str(SMOKE_DIR))

from _antibody_smoke_lib import SCFV_LINKER_LEN, build_scfv_record  # noqa: E402

FIXTURE = SMOKE_DIR / "fixtures" / "demo_scfv.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "smoke_outputs" / "phase_a_cdr"


def _load_fixture(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    rec = build_scfv_record(
        data.get("name", "fixture"), data["vh_seq"], data["vl_seq"]
    )
    L = rec.vh_len + SCFV_LINKER_LEN + rec.vl_len
    struct_seq = data.get("struct_seq") or ",".join(
        str(100 + (i % 8000)) for i in range(L)
    )
    return {
        "name": rec.pdb_id,
        "aa_seq": rec.scfv_sequence,
        "vh_len": rec.vh_len,
        "vl_len": rec.vl_len,
        "struct_seq": struct_seq,
        "cdr_mask": rec.cdr_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Only validate demo fixture and scFv masks",
    )
    args = parser.parse_args()

    fix = _load_fixture(FIXTURE)
    print(
        f"Fixture: {fix['name']}  L={len(fix['aa_seq'])}  "
        f"vh={fix['vh_len']} vl={fix['vl_len']}"
    )
    assert fix["cdr_mask"].sum() > 0
    print(f"CDR positions: {int(fix['cdr_mask'].sum())}")

    if args.skip_gpu:
        print("[PASS] phase_a fixture + masks (CPU-only)")
        return

    gen_script = REPO_ROOT / "generate_antibody.py"
    if not gen_script.exists():
        print(
            "[SKIP] generate_antibody.py not found; run with --skip-gpu "
            "or restore the phase-A generation script.",
            file=sys.stderr,
        )
        sys.exit(0)

    sys.path.insert(0, str(REPO_ROOT / "tests" / "smoke"))
    from common import require_cuda  # noqa: E402

    require_cuda()

    cmd = [
        sys.executable,
        str(gen_script),
        "--fixture",
        str(FIXTURE),
        "--max-iter",
        str(args.max_iter),
        "--num-seqs",
        str(args.num_seqs),
        "--saveto",
        str(args.output),
        "--sampling-strategy",
        "annealing@2.0:0.1",
    ]
    if args.model:
        cmd.extend(["--model-name", str(args.model)])
    if args.struct_tokenizer:
        cmd.extend(["--struct-tokenizer", str(args.struct_tokenizer)])

    env = {
        **__import__("os").environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
    }
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    assert (args.output / "aatype.fasta").exists()
    assert (args.output / "struct_token.fasta").exists()
    metrics_file = args.output / "phase_a_metrics.json"
    assert metrics_file.exists()
    with metrics_file.open() as f:
        metrics = json.load(f)
    print("[PASS] phase_a_cdr_baseline")
    print(f"  output: {args.output}")
    print(f"  metrics: {metrics}")


if __name__ == "__main__":
    main()
