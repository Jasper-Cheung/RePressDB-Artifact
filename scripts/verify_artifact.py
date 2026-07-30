from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            failures.append(relative)

    assessment = json.loads(
        (ROOT / "results" / "frozen_summaries" / "v170_assessment.json").read_text(encoding="utf-8")
    )
    all_ten = bool(assessment.get("acceptance_checks", {}).get("all_ten"))

    print(f"manifest_failures={len(failures)}")
    print(f"v170_all_ten={all_ten}")
    if failures:
        for item in failures:
            print(item)
    return 0 if not failures and all_ten else 1


if __name__ == "__main__":
    raise SystemExit(main())
