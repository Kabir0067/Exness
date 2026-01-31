from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple


class UTCScheduler:
    """Source of Truth for Day/Time logic.

    Weekdays (Mon-Fri): XAU + BTC
    Weekends (Sat-Sun): BTC only
    """

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def is_weekend() -> bool:
        # 5=Sat, 6=Sun
        return UTCScheduler.now_utc().weekday() >= 5

    @staticmethod
    def get_active_assets() -> Tuple[str, ...]:
        if UTCScheduler.is_weekend():
            return ("BTC",)
        return ("XAU", "BTC")

    @staticmethod
    def market_status(asset: str) -> bool:
        """Internal market open check.

        BTC: always open (24/7).
        XAU: weekdays only (hours logic guarded elsewhere by MT5/market feed).
        """
        if asset == "BTC":
            return True
        if asset == "XAU":
            return not UTCScheduler.is_weekend()
        return False
