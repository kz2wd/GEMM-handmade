import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "build"))

import cgemm
cgemm.naive_prepare(10)