"""Pure, offline Modal cost preflight. This does not launch jobs.

Rates are inputs, not trusted constants: verify against https://modal.com/pricing
immediately before a paid run. Estimate is not a provider-side hard spending cap.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CostEstimate:
    gpu_usd_per_second: float
    cpu_cores: float
    memory_gib: float
    seconds: int
    fixed_usd: float = 0.0
    api_usd: float = 0.0

    def total(self) -> float:
        values = (self.gpu_usd_per_second, self.cpu_cores, self.memory_gib,
                  self.seconds, self.fixed_usd, self.api_usd)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Cost estimate inputs must be finite nonnegative numbers")
        return (self.seconds * (self.gpu_usd_per_second
                + self.cpu_cores * 0.0000131
                + self.memory_gib * 0.00000222)
                + self.fixed_usd + self.api_usd)

def approve_preflight(estimate: CostEstimate, *, remaining_usd: float,
                      explicitly_authorized: bool = False) -> float:
    if not explicitly_authorized:
        raise PermissionError("Paid Modal launch requires explicit authorization")
    if not math.isfinite(remaining_usd) or remaining_usd < 0:
        raise ValueError("Invalid remaining budget")
    amount = estimate.total()
    if amount > remaining_usd:
        raise RuntimeError(f"Estimated cost ${amount:.2f} exceeds remaining ${remaining_usd:.2f}")
    return amount
