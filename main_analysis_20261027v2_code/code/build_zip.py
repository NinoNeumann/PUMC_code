"""Create and verify the ZIP64 distribution of the audited main-analysis copy."""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT.with_suffix(".zip")
PARTIAL = ROOT.with_suffix(".zip.partial")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if ZIP.exists() or PARTIAL.exists():
        raise RuntimeError(f"Output exists: {ZIP} / {PARTIAL}")
    started = time.time()
    paths = []
    for base, _, names in os.walk(ROOT):
        for name in names:
            paths.append(Path(base) / name)
    paths.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(PARTIAL, "w", compression=zipfile.ZIP_STORED, allowZip64=True, strict_timestamps=False) as z:
        for i, path in enumerate(paths, 1):
            z.write(path, f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
            if i % 20000 == 0:
                print(f"zipped {i}/{len(paths)} files", flush=True)
    with zipfile.ZipFile(PARTIAL, "r", allowZip64=True) as z:
        infos = z.infolist()
        if len(infos) != len(paths):
            raise RuntimeError(f"ZIP entry count mismatch: {len(infos)} != {len(paths)}")
        for info, path in zip(infos, paths):
            expected_name = f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}"
            if info.filename != expected_name or info.file_size != path.stat().st_size:
                raise RuntimeError(f"ZIP entry mismatch: {info.filename}")
        print("ZIP names and sizes verified; reading every entry for CRC", flush=True)
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failed: {bad}")
    os.replace(PARTIAL, ZIP)
    result = {
        "zip_path": str(ZIP),
        "entries": len(paths),
        "zip_bytes": ZIP.stat().st_size,
        "zip_sha256": sha256(ZIP),
        "all_entries_crc_ok": True,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    ZIP.with_suffix(".zip.audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ZIP.with_suffix(".zip.sha256").write_text(f"{result['zip_sha256']}  {ZIP.name}\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
