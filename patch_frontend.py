"""
Usage (from the project folder):  python3 patch_frontend.py <path-to-your-html-file>
Example:                          python3 patch_frontend.py frontend/index.html

Changes the dashboard page from 5 services to 7 (adds Entertainment and Wifi).
Everything else in the file stays exactly as it is. A backup (.bak) is saved first.
"""
import shutil
import sys
from pathlib import Path

EDITS = [
    # 1. short names used by the sparklines and the flagged-review scores
    ('food__beverages:"Food"};',
     'food__beverages:"Food", inflight_entertainment:"Entertainment", wifi__connectivity:"Wifi"};'),
    # 2. formula shown in the footer
    ('b₅·Food,',
     'b₅·Food + b₆·Entertainment + b₇·Wifi,'),
    # 3. "Five services explain" now follows the real number of services
    ('Five services explain',
     '${S.length} services explain'),
    # 4. flagged reviews explanation
    ('what its five service scores predict',
     'what its seven service scores predict'),
]

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")

    if "wifi__connectivity" in text:
        sys.exit("This file already has the 7-service changes. Nothing to do.")

    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"Could not apply an edit (found {n} matches for: {old!r}). "
                     f"Is this the right file? Nothing was changed.")
        text = text.replace(old, new)

    shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(text, encoding="utf-8")
    print(f"Done. Updated {path} (backup saved as {path.name}.bak)")

if __name__ == "__main__":
    main()