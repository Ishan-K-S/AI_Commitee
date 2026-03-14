from pydantic import BaseModel, Field
from typing import Optional
import time
import threading
from typing import Dict, List


class Filter:
    class Valves(BaseModel):
        REQUESTS_PER_MINUTE: int = Field(
            default=5, description="Max requests per minute per user"
        )

        REQUESTS_PER_DAY: int = Field(
            default=120, description="Max requests per day per user"
        )

        REQUESTS_PER_WEEK: int = Field(
            default=480, description="Max requests per week per user"
        )

    def __init__(self):
        self.id = "user_rate_limit"
        self.name = "User Rate Limit Filter"
        self.valves = self.Valves()

        self._rate_log: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, user_id: str, now: float):
        """Remove timestamps older than 1 week. Must be called with self._lock held."""
        timestamps = self._rate_log.get(user_id)

        if not timestamps:
            return

        cleaned = [t for t in timestamps if now - t < 60 * 60 * 24 * 7]

        # cleaned up empty keys to prevent too much memory be used
        if cleaned:
            self._rate_log[user_id] = cleaned
        else:
            del self._rate_log[user_id]

    def _check_limit(self, user_id: str) -> Optional[str]:
        now = time.time()

        self._prune(user_id, now)

        timestamps = self._rate_log.get(user_id, [])

        last_minute = sum(1 for t in timestamps if now - t < 60)
        last_day = sum(1 for t in timestamps if now - t < 60 * 60 * 24)
        last_week = sum(1 for t in timestamps if now - t < 60 * 60 * 24 * 7)

        if last_minute >= self.valves.REQUESTS_PER_MINUTE:
            return "Rate limit exceeded. Try again in a minute."

        if last_day >= self.valves.REQUESTS_PER_DAY:
            return "Daily rate limit exceeded."

        if last_week >= self.valves.REQUESTS_PER_WEEK:
            return "Weekly rate limit exceeded."

        self._rate_log.setdefault(user_id, []).append(now)

        return None

    def inlet(self, body: dict, __user__: Optional[dict] = None):
        """
        Runs before the pipe executes.
        """
        user_info = __user__ or {}
        user_id = user_info.get("id")

        # instead of calling all anonymous users "anonymous", try to find something unique about them like session id or ip
        if not user_id:
            user_id = user_info.get("session_id") or user_info.get("ip") or "anonymous"

        # make sure lock checks entire sequence
        with self._lock:
            err = self._check_limit(user_id)

        if err:
            raise Exception(err)

        return body
