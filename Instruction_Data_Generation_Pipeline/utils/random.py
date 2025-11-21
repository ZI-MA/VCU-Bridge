import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Set global random seed for best-effort reproducibility.

    Note: PYTHONHASHSEED is read at interpreter start; setting it here won't
    affect current process hashing, but we set it for subprocesses/logging clarity.
    """
    if seed is None:
        return
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
    except Exception:
        pass


def derive_int_seed(*values: object, base: int = 0) -> int:
    """Derive a stable 32-bit seed from inputs.

    Combines a base seed with the hashes of provided values to get a deterministic seed
    suitable for initializing random.Random without sharing state across threads.
    """
    acc = base & 0xFFFFFFFF
    for v in values:
        h = hash(v) & 0xFFFFFFFF
        # Xorshift-like mixing
        acc ^= (h + 0x9E3779B9 + ((acc << 6) & 0xFFFFFFFF) + (acc >> 2)) & 0xFFFFFFFF
    return acc & 0xFFFFFFFF

