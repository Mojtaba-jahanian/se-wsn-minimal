import numpy as np

def inject_burst_failures(alive: np.ndarray, failure_rate_percent: float, rng: np.random.Generator) -> np.ndarray:
    alive_idx = np.flatnonzero(alive)
    k = int(round(len(alive_idx) * failure_rate_percent / 100.0))
    out = alive.copy()
    if k > 0 and len(alive_idx) > 0:
        dropped = rng.choice(alive_idx, size=min(k, len(alive_idx)), replace=False)
        out[dropped] = False
    return out
