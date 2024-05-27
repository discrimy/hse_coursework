"""Утилиты для работы с логами"""

import functools
import logging
from typing import TypeVar, Callable, Any


_logger = logging.getLogger("disabled")


C = TypeVar("C", bound=Callable)  # noqa VNE001


def disable_with_logging(enabled: bool) -> Callable[[C], C]:
    """'Выключает' функцию, заменяя её вызов на логгирование вызова"""

    def decorator(func: C) -> C:
        if enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:  # type: ignore
            args_formatter = [repr(arg) for arg in args]
            kwargs_formatter = [f"{key}={value!r}" for key, value in kwargs.items()]
            func_full_name = f"{func.__module__}.{func.__name__}"
            _logger.debug(
                "Вызов выключенной функции: %s(%s)",
                func_full_name,
                ", ".join(args_formatter + kwargs_formatter),
            )

        return wrapper  # type: ignore

    return decorator
