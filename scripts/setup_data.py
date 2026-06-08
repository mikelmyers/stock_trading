"""Restore local datasets from the repo's bundled archives.

Usage:
    python scripts/setup_data.py              # extract EODHD zip if missing
    python scripts/setup_data.py --force      # re-extract EODHD
    python scripts/setup_data.py --from-downloads
        # copy from %%USERPROFILE%%\\Downloads\\archive* if present
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EODHD_ZIP = DATA_DIR / "eodhd.zip"
EODHD_DIR = DATA_DIR / "eodhd"
ARANDKEI_DIR = DATA_DIR / "arandkei" / "delisted"

DEFAULT_DOWNLOADS = [
    Path.home() / "Downloads" / "archive",
    Path.home() / "Downloads" / "archive (1)",
]


def _extract_eodhd(force: bool = False) -> None:
    if not EODHD_ZIP.exists():
        raise FileNotFoundError(f"Missing archive: {EODHD_ZIP}")

    marker = EODHD_DIR / ".extracted"
    if marker.exists() and not force:
        print(f"EODHD already extracted at {EODHD_DIR} (use --force to redo)")
        return

    if EODHD_DIR.exists() and force:
        shutil.rmtree(EODHD_DIR)

    EODHD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {EODHD_ZIP.name} -> {EODHD_DIR} ...")
    with zipfile.ZipFile(EODHD_ZIP) as zf:
        zf.extractall(EODHD_DIR)
    marker.write_text("ok\n", encoding="utf-8")
    stocks = len(list((EODHD_DIR / "Stocks").glob("*.us.txt"))) if (EODHD_DIR / "Stocks").exists() else 0
    etfs = len(list((EODHD_DIR / "ETFs").glob("*.us.txt"))) if (EODHD_DIR / "ETFs").exists() else 0
    print(f"  Done: {stocks} stock files, {etfs} ETF files")


def _copy_from_downloads() -> None:
    archive = DEFAULT_DOWNLOADS[0]
    archive1 = DEFAULT_DOWNLOADS[1]

    if archive.exists():
        for sub in ("Stocks", "ETFs"):
            src = archive / sub
            if not src.exists():
                continue
            dst = EODHD_DIR / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.us.txt"):
                target = dst / f.name
                if not target.exists():
                    shutil.copy2(f, target)
        print(f"Synced EODHD files from {archive}")

    if archive1.exists():
        ARANDKEI_DIR.mkdir(parents=True, exist_ok=True)
        for f in archive1.glob("*.csv"):
            target = ARANDKEI_DIR / f.name
            if not target.exists():
                shutil.copy2(f, target)
        readme = archive1 / "README.md"
        if readme.exists():
            shutil.copy2(readme, ARANDKEI_DIR / "README.md")
        print(f"Synced Arandkei files from {archive1}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore datasets for stock_trading")
    parser.add_argument("--force", action="store_true", help="Re-extract EODHD zip")
    parser.add_argument(
        "--from-downloads",
        action="store_true",
        help="Copy missing files from Downloads/archive folders",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_downloads:
        _copy_from_downloads()

    _extract_eodhd(force=args.force)
    print(f"\nData ready under {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())