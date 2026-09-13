# app/utils/rate_limit_utils.py
import time
from typing import Deque
from app.core.logger import logger  # Reuse the project-wide logger.


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    """
    Generic sliding-window API rate limiter.
    Maintains a deque of request timestamps and waits automatically when the
    number of requests inside the window reaches the limit.
    :param request_times: Externally initialized deque of request timestamps.
    :param max_requests: Maximum allowed requests in the rate-limit window.
    :param window_seconds: Sliding-window duration in seconds. Defaults to 60.
    :return: None. Blocks until the request is allowed when the limit is reached.
    """
    current_time = time.time()

    # 1. Remove expired request timestamps outside the sliding window.
    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    # 2. Wait for the remaining window time when the limit is reached.
    if len(request_times) >= max_requests:
        # Remaining wait time = window duration - age of the oldest request.
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(
                f"API rate limit reached: max {max_requests} requests per {window_seconds}s window. "
                f"Waiting {sleep_duration:.2f}s."
            )
            time.sleep(sleep_duration)
            # Refresh the timestamp and remove requests that expired while waiting.
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    # 3. Record the current request timestamp.
    request_times.append(current_time)
    logger.debug(f"API request timestamp recorded. Requests in the current {window_seconds}s window: {len(request_times)}")
