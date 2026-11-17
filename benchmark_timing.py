"""Shared wall clock for model timings; waits for queued CUDA work so GPU times are not understated."""
import time


def clock() -> float:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
