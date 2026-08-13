from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(
    os.environ.get("LGAIMERS_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()
DATA_DIR: Final = os.fspath(
    Path(os.environ.get("LGAIMERS_DATA_DIR", PROJECT_ROOT / "데이터" / "open" / "data"))
) + os.sep
EXPERIMENTS_DIR: Final = Path(
    os.environ.get("LGAIMERS_EXPERIMENTS_DIR", PROJECT_ROOT / "experiments")
).expanduser().resolve()
TMP_DATA_DIR: Final = os.fspath(
    Path(os.environ.get("LGAIMERS_TMP_DATA_DIR", EXPERIMENTS_DIR / "tmpdata"))
) + os.sep
KBO_INSIGHTS_DIR: Final = os.fspath(
    Path(os.environ.get("LGAIMERS_KBO_INSIGHTS_DIR", EXPERIMENTS_DIR / "kbo_insights_out"))
)
