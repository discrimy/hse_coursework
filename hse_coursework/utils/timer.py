import datetime
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Protocol, Any, Optional, Generator

_T = TypeVar("_T")


class SupportsSub(Protocol):
    """Протокол для объектов с поддержкой метода `__sub__`"""

    def __sub__(self, other) -> Any: ...


@dataclass
class Measure(Generic[_T]):
    """
    Класс для измерения изменения различных величин (время, количество запросов к БД и т.д.)

    >>> measure: Measure[datetime.timedelta] = Measure(start=datetime.datetime.now())
    >>> # do a stuff
    >>> measure.end = datetime.datetime.now()
    >>> measure.delta  # Время, потраченное на выполнение
    >>> # Из-за аннотации на уровне объявления переменной, статические анализаторы поймут что тип - datetime.timedelta;
    """

    start: SupportsSub
    end: Optional[SupportsSub] = field(init=False, default=None)

    @property
    def delta(self) -> _T:
        """Дельта между началом и концом"""
        if self.end is None:
            raise ValueError("Измерение ещё не окончено")

        return self.end - self.start


@contextmanager
def measure_time() -> Generator[Measure[datetime.timedelta], None, None]:
    """
    Измеряет время, потраченное на выполнение кода внутри менеджера.

    >>> with measure_time() as measure:
    ...     pass  # измеряемый блок кода
    >>> measure.delta  # type: datetime.timedelta; время, потраченное на выполнение
    """
    measure = Measure[datetime.timedelta](start=datetime.datetime.now())

    yield measure

    measure.end = datetime.datetime.now()


@contextmanager
def measure_performance() -> Generator[Measure[float], None, None]:
    """
    Измеряет производительность кода (затраченное время) блока под контекстным менеджером.
    Для измерения используется `time.perf_count()`.
    Почему не стоит измерять производительность с применением `datetime.datetime` или `time.time`
    можно почитать здесь: https://www.raaicode.com/using-time-monotonic-and-perf_counter-to-measure-time-in-python/

    Пример использования
    ----
    >>> with measure_performance() as measure:
    ...     pass  # измеряемый блок кода
    >>> measure.delta  # type: float; количество секунд, потраченных на выполнение
    """
    measure = Measure[float](start=time.perf_counter())

    yield measure

    measure.end = time.perf_counter()
