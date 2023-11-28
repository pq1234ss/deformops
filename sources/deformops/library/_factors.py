r"""Utility for finding factors of a number."""

__all__ = []


def find_factors(n: int) -> list[int]:
    r"""
    Simple utility to find factors of a number.
    """
    if n < 0:
        msg = f"Expected a positive integer. Got: {n}!"
        raise ValueError(msg)
    result: set[int] = set()
    for i in range(1, int(n**0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    return sorted(result)
