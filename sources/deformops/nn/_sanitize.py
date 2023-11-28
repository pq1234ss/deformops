"""Utility functions for input sanitization."""

from typing import NoReturn

__all__ = []


def CHECK_2POWER(n: float) -> None | NoReturn:
    r"""Check whether :param:`n` is a positive power of 2."""

    if (not isinstance(n, int)) or (n < 0):
        msg = f"Expected a positive integer. Got: {n} ({type(n)})!"
        raise ValueError(msg)
    is_2power = (n & (n - 1) == 0) and n != 0
    if not is_2power:
        msg = f"Expected a power of 2. Got: {n}!"
        raise AssertionError(msg)
    return None


def CHECK_DIVISIBLE(a: int, b: int) -> None | NoReturn:
    r"""Check whether :param:`a` is divisible by :param:`b`."""
    is_divisible = a % b == 0
    if not is_divisible:
        msg = f"Expected {a} to be divisible by {b}!"
        raise AssertionError(msg)
    return None
