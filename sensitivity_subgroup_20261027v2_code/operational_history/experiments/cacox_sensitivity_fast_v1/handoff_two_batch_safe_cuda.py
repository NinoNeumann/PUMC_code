"""Handoff after 96% RAM use: keep six MICE workers, cap CUDA selection at two per batch."""
from __future__ import annotations

import full_controller as frozen
import handoff_two_batch_phase_split as transition


transition.OLD_CONTROLLER = 33216
transition.OLD_BATCHES = {2: 11756, 3: 52688}
transition.IN_FLIGHT = (("egfr_observed", 13), ("renal_observed", 22))
transition.AUDIT = frozen.RUN / "TWO_BATCH_SAFE_CUDA_HANDOFF.json"
transition.ARCHIVE = frozen.RUN / "handoffs/two_batch_safe_cuda_20261107_v1"
transition.REASON = (
    "Two simultaneous CUDA selections reached 96.162% system RAM use; "
    "retain six MICE workers but cap selection at two per batch"
)


if __name__ == "__main__":
    try:
        transition.main()
    except BaseException as exc:
        transition.record("failed", error=repr(exc))
        raise
