import torch.cuda


def measure_speed(func, inputs, *, warmup: int, iterations: int) -> float:
    """
    Utility for measuring execution time in ms.
    """
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        func(*inputs)

    torch.cuda.synchronize()
    tic.record()

    for _ in range(iterations):
        func(*inputs)

    toc.record()
    torch.cuda.synchronize()

    return tic.elapsed_time(toc) / iterations
