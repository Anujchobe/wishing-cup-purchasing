"""Boston time. Every business date in the app (today, week starts, needed-by
dates) is computed in America/New_York.

Timestamps are stored as UTC and read back in Boston time (see
app.models.UtcDateTime), so rows never sort wrongly across daylight-saving
changes. On Windows, ZoneInfo needs the tzdata package (in requirements.txt).
"""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

BOSTON = ZoneInfo("America/New_York")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def now() -> datetime:
    return datetime.now(BOSTON)


def today() -> date:
    return now().date()


def at_boston(day: date, at: time = time(0)) -> datetime:
    """A Boston wall-clock time on a given day, timezone-aware."""
    return datetime.combine(day, at, tzinfo=BOSTON)


def to_boston(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive datetime: attach a timezone before converting")
    return dt.astimezone(BOSTON)


def week_start(day: date) -> date:
    """Monday of the week that contains `day`."""
    return day - timedelta(days=day.weekday())


def weekday_name(day: date) -> str:
    return WEEKDAYS[day.weekday()]
