"""Download USTC-TFC2016 from GitHub and lay it out as raw_dir/<Group>/<Class>/*.pcap.

Plain .pcap files are copied; .7z archives are extracted (an archive may contain
several pcap parts of the same class). The class name is the file/archive stem.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import py7zr

from etc.config import load_config, load_dataset_config, resolve_path
from etc.utils import get_logger

REPO_URL = "https://github.com/yungshenglu/USTC-TFC2016.git"
PCAP_SUFFIXES = {".pcap", ".pcapng"}

log = get_logger()


def safe_extract_7z(archive: Path, dest: Path) -> list[Path]:
    """Extract only pcap members of `archive` into `dest`, refusing path traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        names = [n for n in z.getnames() if Path(n).suffix.lower() in PCAP_SUFFIXES]
        for n in names:
            if Path(n).is_absolute() or ".." in Path(n).parts:
                raise ValueError(f"Unsafe member path in {archive}: {n}")
        z.extract(path=dest, targets=names)
    # Flatten any internal folders so all pcaps sit directly in dest.
    out = []
    for n in names:
        src = dest / n
        target = dest / Path(n).name
        if src != target:
            shutil.move(src, target)
        out.append(target)
    for d in sorted(dest.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", default="data/raw/_download")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(load_dataset_config(cfg)["raw_dir"])
    clone_dir = resolve_path(args.download_dir) / "USTC-TFC2016"

    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Cloning {REPO_URL} -> {clone_dir}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)], check=True)

    for group in ("Benign", "Malware"):
        for f in sorted((clone_dir / group).iterdir()):
            cls_dir = raw_dir / group / f.stem
            if cls_dir.exists() and any(cls_dir.iterdir()):
                log.info(f"skip {group}/{f.stem} (already present)")
                continue
            cls_dir.mkdir(parents=True, exist_ok=True)
            if f.suffix.lower() in PCAP_SUFFIXES:
                shutil.copy2(f, cls_dir / f.name)
                files = [cls_dir / f.name]
            elif f.suffix.lower() == ".7z":
                files = safe_extract_7z(f, cls_dir)
            else:
                continue
            size_mb = sum(p.stat().st_size for p in files) / 1e6
            log.info(f"{group}/{f.stem}: {len(files)} file(s), {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
