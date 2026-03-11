from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str):
    return logger.bind(module=name)


def timed(func: F) -> F:
    """Decorator that logs execution time of sync and async functions."""
    if _is_async(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__qualname__} completed in {elapsed:.3f}s")
            return result
        return async_wrapper  # type: ignore[return-value]
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__qualname__} completed in {elapsed:.3f}s")
            return result
        return sync_wrapper  # type: ignore[return-value]


def _is_async(func: Callable[..., Any]) -> bool:
    import asyncio
    import inspect
    return asyncio.iscoroutinefunction(func) or inspect.iscoroutinefunction(func)
