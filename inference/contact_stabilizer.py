"""Confirm a future contact across native-rate CAD-WM forecasts."""

import numpy as np


class FutureContactStabilizer:
    """Filter teacher inputs without changing its prefix or fusion implementation.

    Entry requires consecutive forecasts agreeing on an overlapping target
    time. During exit confirmation, retain the last accepted forecast instead
    of opening fusion on a newly suppressed/free wrench window.
    """

    def __init__(self, *, enter_frames=2, exit_frames=1):
        self.enter_frames = int(enter_frames)
        self.exit_frames = int(exit_frames)
        if self.enter_frames < 1 or self.exit_frames < 1:
            raise ValueError("future contact confirmation frame counts must be positive")
        self.reset()

    def reset(self):
        self.contact = False
        self._exit_count = 0
        self._confirmation_counts = None
        self._accepted_future = None

    def update(self, future, contact_mask):
        mask = np.asarray(contact_mask, dtype=bool)
        raw_contact = bool(mask.any())
        # Each count follows one absolute target time as it moves left through
        # the horizon. This also requires genuine N-frame agreement when N > 2.
        previous = self._confirmation_counts
        aligned = np.zeros(mask.shape, dtype=np.int64)
        if previous is not None:
            aligned[:-1] = previous[1:]
        self._confirmation_counts = np.where(mask, aligned + 1, 0)
        if self.contact:
            self._exit_count = 0 if raw_contact else self._exit_count + 1
            if self._exit_count >= self.exit_frames:
                self.contact = False
                self._accepted_future = None
        elif raw_contact:
            if self._confirmation_counts.max() >= self.enter_frames:
                self.contact = True
                self._exit_count = 0
        if self.contact and raw_contact:
            self._accepted_future = future.copy()
        filtered = self._accepted_future.copy() if self.contact else np.zeros_like(future)
        return filtered, {
            "raw_has_contact": raw_contact,
            "has_contact": self.contact,
            "holding": self.contact and not raw_contact,
        }
