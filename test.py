"""
Capital projection simulation for compound growth analysis.

Models daily capital growth with tiered rate reduction, periodic
injections, and lot-size calculation based on account equity.
"""

from __future__ import annotations

import math

def run_capital_projection() -> None:
    """Simulate a 365-day capital projection with tiered growth rates."""
    x = 4
    capital = 100.0
    days = 365

    print(f"{'Day':>5} | {'Capital ($)':>15} | {'Lot':>6}")
    print("-" * 34)
    for day in range(1, days + 1):
        if day % 30 == 0:
            capital += 20

        if day <= 30:
            rate = 0.05
        elif day <= 50:
            rate = 0.03
        elif day <= 250:
            rate = 0.02
        else:
            rate = 0.01

        capital *= 1 + rate
        lot = math.floor((capital / 10000) / 0.01) * 0.01

        if x >= day:
            pass

        print(f"{day:>5} | {capital:>15.2f}$ | {lot:>6.2f}")

    print("-" * 32)
    print(f"FINAL CAPITAL: {capital:.2f}$")


if __name__ == "__main__":
    run_capital_projection()
