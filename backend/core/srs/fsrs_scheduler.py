from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

try:
    from fsrs import Card, FSRS as FsrsScheduler, Rating, State
except ImportError:  # pragma: no cover - depends on fsrs package version
    from fsrs import Card, Rating, Scheduler as FsrsScheduler, State


def _parse_datetime(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_state(value: Any) -> State | None:
    if value is None:
        return None
    try:
        return State[str(value)]
    except (KeyError, ValueError):
        return None


def card_from_dict(data: dict[str, Any]) -> Card:
    card = Card()
    if not data:
        return card

    due = _parse_datetime(data.get("due"))
    try:
        card.due = due
    except AttributeError:
        pass

    for field in ("stability", "difficulty", "elapsed_days", "scheduled_days", "reps", "lapses"):
        if field in data:
            try:
                setattr(card, field, data[field])
            except AttributeError:
                pass

    state = _parse_state(data.get("state"))
    if state is not None:
        try:
            card.state = state
        except AttributeError:
            pass

    return card


def _normalize_card(card: Card) -> None:
    # FSRS expects stability to be positive for recall transitions.
    stability = getattr(card, "stability", None)
    if stability is None or stability <= 0:
        try:
            card.stability = 0.1
        except AttributeError:
            pass

    difficulty = getattr(card, "difficulty", None)
    if difficulty is None or difficulty <= 0:
        try:
            card.difficulty = 5.0
        except AttributeError:
            pass


def card_to_dict(card: Card) -> dict[str, Any]:
    return {
        "due": getattr(card, "due", datetime.now(timezone.utc)).isoformat(),
        "stability": float(getattr(card, "stability", 0.0) or 0.0),
        "difficulty": float(getattr(card, "difficulty", 0.0) or 0.0),
        "elapsed_days": int(getattr(card, "elapsed_days", 0) or 0),
        "scheduled_days": int(getattr(card, "scheduled_days", 0) or 0),
        "reps": int(getattr(card, "reps", 0) or 0),
        "lapses": int(getattr(card, "lapses", 0) or 0),
        "state": getattr(getattr(card, "state", ""), "name", "New"),
    }


def new_card_dict() -> dict[str, Any]:
    return card_to_dict(Card())


def _rating_from_names(preferred: list[str], fallback_index: int) -> Rating:
    for name in preferred:
        if hasattr(Rating, name):
            return getattr(Rating, name)

    ratings = list(Rating)
    if not ratings:
        raise RuntimeError("Unsupported fsrs Rating enum")

    index = max(0, min(fallback_index, len(ratings) - 1))
    return ratings[index]


def map_rating(score: int) -> Rating:
    if score <= 1:
        return _rating_from_names(["Again", "Fail", "Forgot"], 0)
    if score == 2:
        return _rating_from_names(["Hard"], 1)
    if score == 3:
        return _rating_from_names(["Good", "Okay", "Fair"], 2)
    return _rating_from_names(["Easy", "Excellent"], len(list(Rating)) - 1)


def review_card(card_dict: dict[str, Any], score: int, now: datetime | None = None) -> dict[str, Any]:
    scheduler = FsrsScheduler()
    card = card_from_dict(card_dict)
    _normalize_card(card)
    rating = map_rating(score)
    review_time = now or datetime.now(timezone.utc)

    if hasattr(scheduler, "review"):
        try:
            new_card, _ = scheduler.review(card, rating, review_time)
        except TypeError:
            new_card, _ = scheduler.review(card, rating)
    elif hasattr(scheduler, "review_card"):
        try:
            new_card, _ = scheduler.review_card(card, rating, review_time)
        except TypeError:
            new_card, _ = scheduler.review_card(card, rating)
    else:
        raise RuntimeError("Unsupported fsrs scheduler API")

    return card_to_dict(new_card)


def due_date(card_dict: dict[str, Any]) -> date:
    return _parse_datetime(card_dict.get("due")).date()


def is_due(card_dict: dict[str, Any], on_date: date | None = None) -> bool:
    today = on_date or datetime.now(timezone.utc).date()
    return due_date(card_dict) <= today
