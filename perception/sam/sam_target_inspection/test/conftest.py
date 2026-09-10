"""Path setup for the off-vehicle tests.

`sam_target_inspection` imports `sam_farm_inspection.change_point` and
`sam_farm_inspection.farm_ledger` — deliberately, because a second copy of the change-point
maths or of the association rules is a second thing to keep right (SETTLED §3d). Off the
vehicle there is no ament index, so both package roots go on the path here.
"""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FARM = PKG.parent / "sam_farm_inspection"
for p in (str(PKG), str(FARM)):
    if p not in sys.path:
        sys.path.insert(0, p)
