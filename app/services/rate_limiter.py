"""
Centralized rate limiter for Replicate API calls.

This module ensures the system respects Replicate's rate limits and usage policies:
- Maximum 600 requests per minute (staying safely below limit)
- Exponential backoff on 429 errors
- Minimum 30 second wait after rate limit hit
- Global coordination across all workers
"""

import logging
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ReplicateRateLimiter:
    """
    Thread-safe rate limiter for Replicate API calls.
    
    Implements token bucket algorithm with:
    - Maximum 600 requests per minute (10 per second)
    - Burst capacity of 20 requests
    - Automatic rate reduction after 429 errors
    - Exponential backoff on repeated failures
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 5,  # Credit-only tier: max 6/min, stay at 5 to be safe
        burst_capacity: int = 3,  # Very conservative burst for credit tier
        min_interval_seconds: float = 1.2,  # 1.2s minimum between requests (safer than 1.0s)
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute (default: 5 for credit-only tier)
            burst_capacity: Maximum burst size (default: 3 for credit-only tier)
            min_interval_seconds: Minimum time between requests (default: 1.2s for credit-only tier)
        
        Note: These defaults are configured for Replicate's credit-only tier limits:
        - Credit-only tier: 1 request/second, max 6 requests/minute
        - We use 5 requests/minute to stay safely below the limit
        - Minimum 1.2s between requests to avoid hitting the 1 req/sec limit
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.burst_capacity = burst_capacity
        self.min_interval_seconds = min_interval_seconds
        
        # Token bucket state
        self.tokens = float(burst_capacity)
        self.max_tokens = float(burst_capacity)
        self.refill_rate = max_requests_per_minute / 60.0  # Tokens per second
        self.last_refill = time.time()
        
        # Request tracking
        self.request_times: deque = deque(maxlen=max_requests_per_minute)
        self.last_request_time = 0.0
        
        # Rate limit backoff state
        self.rate_limited_until: Optional[float] = None
        self.consecutive_429s = 0
        self.backoff_multiplier = 1.0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(
            f"Replicate rate limiter initialized: "
            f"{max_requests_per_minute} req/min, "
            f"burst: {burst_capacity}, "
            f"min interval: {min_interval_seconds}s"
        )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time (token bucket algorithm)."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def _is_rate_limited(self) -> bool:
        """Check if we're currently in a rate limit backoff period."""
        if self.rate_limited_until is None:
            return False
        
        if time.time() < self.rate_limited_until:
            return True
        
        # Backoff period expired, reset state
        self.rate_limited_until = None
        self.consecutive_429s = 0
        self.backoff_multiplier = 1.0
        logger.info("Rate limit backoff period expired, resuming normal operation")
        return False
    
    def _calculate_backoff(self) -> float:
        """
        Calculate exponential backoff duration after 429 error.
        
        Returns:
            Backoff duration in seconds
        """
        # Base backoff: 30 seconds (minimum required)
        base_backoff = 30.0
        
        # Exponential multiplier: 2^(consecutive_429s - 1)
        # First 429: 30s, Second: 60s, Third: 120s, etc.
        exponential_factor = 2 ** (self.consecutive_429s - 1)
        
        # Cap at 5 minutes to avoid excessive waits
        backoff = min(base_backoff * exponential_factor, 300.0)
        
        return backoff
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a Replicate API call.
        
        This method blocks until a token is available or timeout is reached.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            True if permission granted, False if timeout reached
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Check if we're in rate limit backoff
                if self._is_rate_limited():
                    wait_time = self.rate_limited_until - time.time()
                    logger.warning(
                        f"Rate limited: waiting {wait_time:.1f}s before next request "
                        f"(consecutive 429s: {self.consecutive_429s})"
                    )
                    
                    # Check timeout
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            logger.error("Rate limiter timeout reached")
                            return False
                    
                    # Release lock and sleep
                    time.sleep(min(1.0, wait_time))
                    continue
                
                # Refill tokens
                self._refill_tokens()
                
                # Check if we have tokens available
                if self.tokens >= 1.0:
                    # Enforce minimum interval between requests
                    now = time.time()
                    time_since_last = now - self.last_request_time
                    
                    if time_since_last < self.min_interval_seconds:
                        sleep_time = self.min_interval_seconds - time_since_last
                        time.sleep(sleep_time)
                        now = time.time()
                    
                    # Consume token
                    self.tokens -= 1.0
                    self.last_request_time = now
                    self.request_times.append(now)
                    
                    logger.debug(
                        f"Rate limiter: token acquired "
                        f"(tokens remaining: {self.tokens:.1f}/{self.max_tokens})"
                    )
                    return True
                
                # No tokens available, check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.error("Rate limiter timeout reached (no tokens)")
                        return False
            
            # Wait a bit before retrying (outside lock)
            time.sleep(0.1)
    
    def report_429(self) -> None:
        """
        Report that a 429 (Too Many Requests) error was received.
        
        This triggers exponential backoff and reduces request rate.
        """
        with self.lock:
            self.consecutive_429s += 1
            backoff = self._calculate_backoff()
            self.rate_limited_until = time.time() + backoff
            
            # Reduce burst capacity temporarily
            self.backoff_multiplier = max(0.5, self.backoff_multiplier * 0.8)
            self.max_tokens = self.burst_capacity * self.backoff_multiplier
            self.tokens = min(self.tokens, self.max_tokens)
            
            logger.error(
                f"Replicate 429 error (consecutive: {self.consecutive_429s}). "
                f"Backing off for {backoff:.1f}s. "
                f"Reduced burst capacity to {self.max_tokens:.1f}"
            )
    
    def report_success(self) -> None:
        """
        Report that a request succeeded.
        
        This gradually restores normal rate limits after 429 errors.
        """
        with self.lock:
            if self.consecutive_429s > 0:
                # Gradually restore capacity
                self.backoff_multiplier = min(1.0, self.backoff_multiplier * 1.1)
                self.max_tokens = self.burst_capacity * self.backoff_multiplier
                
                # Reset consecutive 429 counter after successful request
                if self.consecutive_429s > 0:
                    self.consecutive_429s = max(0, self.consecutive_429s - 1)
                    logger.info(
                        f"Request succeeded, reducing 429 counter to {self.consecutive_429s}"
                    )
    
    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dictionary with current state
        """
        with self.lock:
            now = time.time()
            
            # Calculate requests in last minute
            cutoff = now - 60.0
            recent_requests = sum(1 for t in self.request_times if t > cutoff)
            
            return {
                "tokens_available": self.tokens,
                "max_tokens": self.max_tokens,
                "requests_last_minute": recent_requests,
                "max_requests_per_minute": self.max_requests_per_minute,
                "is_rate_limited": self._is_rate_limited(),
                "consecutive_429s": self.consecutive_429s,
                "backoff_multiplier": self.backoff_multiplier,
                "rate_limited_until": (
                    datetime.fromtimestamp(self.rate_limited_until).isoformat()
                    if self.rate_limited_until
                    else None
                ),
            }


# Global rate limiter instance
_rate_limiter: Optional[ReplicateRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> ReplicateRateLimiter:
    """
    Get or create the global Replicate rate limiter instance.
    
    Configured for Replicate's credit-only tier limits:
    - 1 request per second
    - Maximum 6 requests per minute
    - We use 5 requests/minute to stay safely below limit
    
    Returns:
        Global ReplicateRateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = ReplicateRateLimiter(
                    max_requests_per_minute=5,  # Credit-only tier: stay below 6/min limit
                    burst_capacity=3,  # Very conservative burst
                    min_interval_seconds=1.2,  # 1.2s minimum (safer than 1.0s)
                )
    
    return _rate_limiter


def acquire_replicate_token(timeout: Optional[float] = 30.0) -> bool:
    """
    Acquire permission to make a Replicate API call.
    
    This is the main entry point for all Replicate API calls.
    
    Args:
        timeout: Maximum time to wait in seconds (default: 30s)
    
    Returns:
        True if permission granted, False if timeout reached
    """
    limiter = get_rate_limiter()
    return limiter.acquire(timeout=timeout)


def report_replicate_429() -> None:
    """
    Report that a 429 (Too Many Requests) error was received from Replicate.
    
    This triggers exponential backoff and reduces request rate.
    """
    limiter = get_rate_limiter()
    limiter.report_429()


def report_replicate_success() -> None:
    """
    Report that a Replicate request succeeded.
    
    This gradually restores normal rate limits after 429 errors.
    """
    limiter = get_rate_limiter()
    limiter.report_success()


def get_rate_limiter_stats() -> dict:
    """
    Get current rate limiter statistics.
    
    Returns:
        Dictionary with current state
    """
    limiter = get_rate_limiter()
    return limiter.get_stats()
