"""Automatic data fetching from public Google Drive (no login required).

The raw WiDS Datathon 2020 files are not stored in the Git repository. They are
downloaded on demand from public Google Drive links so the application can run
end-to-end inside a container without any local files or user accounts.
"""
from __future__ import annotations

from pathlib import Path

import gdown


ROOT = Path(__file__).resolve().parents[1]

# Public Google Drive file IDs (shared as "anyone with the link").
DRIVE_FILES: dict[str, str] = {
    "data/raw/training_v2.csv": "1WzcyMe2HlTjmFnq8uPFxrb-KG9lx2KfA",
    "data/raw/unlabeled.csv": "1Vdo6P2zOEKgKhXxrRnXXMJEO7JOgLaVh",
}


def ensure_data(force: bool = False) -> None:
    """Download required raw data files if they are not present locally.

    Parameters
    ----------
    force:
        Re-download even if the file already exists.
    """
    for rel_path, file_id in DRIVE_FILES.items():
        path = ROOT / rel_path
        if path.exists() and not force:
            print(f"[data_fetch] already present: {rel_path}")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data_fetch] downloading: {rel_path}")
        gdown.download(id=file_id, output=str(path), quiet=False)


if __name__ == "__main__":
    ensure_data()
    print("[data_fetch] all raw data ready.")
