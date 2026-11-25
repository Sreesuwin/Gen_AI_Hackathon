"""Recommendation engine and optimization simulators (M3)."""

from src.recommend.rules import generate_recommendations
from src.recommend.simulator import (
    simulate_battery,
    simulate_ev,
    simulate_extra_pv,
    simulate_load_shift,
)

__all__ = [
    "simulate_battery",
    "simulate_extra_pv",
    "simulate_load_shift",
    "simulate_ev",
    "generate_recommendations",
]
