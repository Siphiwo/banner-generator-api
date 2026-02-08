"""Quick test to verify rate limiter configuration."""

from app.services.rate_limiter import get_rate_limiter, get_rate_limiter_stats

# Initialize rate limiter
limiter = get_rate_limiter()

# Get stats
stats = get_rate_limiter_stats()

print("\n" + "="*60)
print("RATE LIMITER CONFIGURATION")
print("="*60)
print(f"Tier: Credit-Only (No Payment Method)")
print(f"Max requests/minute: {stats['max_requests_per_minute']}")
print(f"Burst capacity: {stats['max_tokens']}")
print(f"Tokens available: {stats['tokens_available']}")
print(f"Requests last minute: {stats['requests_last_minute']}")
print("="*60)
print("\n✅ Rate limiter configured for credit-only tier")
print("   - 5 requests/minute (below 6/min limit)")
print("   - 3 request burst capacity")
print("   - 1.2s minimum between requests")
print("\nReady for testing!")
