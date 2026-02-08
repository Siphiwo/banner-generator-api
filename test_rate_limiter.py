"""
Test script for Replicate rate limiter with credit-only tier limits.

This script tests the rate limiter configured for:
- 1 request per second
- Maximum 5 requests per minute (staying below 6/min limit)
- Minimum 1.2s between requests
"""

import time
from datetime import datetime
from app.services.rate_limiter import get_rate_limiter, get_rate_limiter_stats


def print_stats():
    """Print current rate limiter statistics."""
    stats = get_rate_limiter_stats()
    print(f"\n📊 Rate Limiter Stats:")
    print(f"   Tokens available: {stats['tokens_available']:.2f}/{stats['max_tokens']:.0f}")
    print(f"   Requests last minute: {stats['requests_last_minute']}")
    print(f"   Max requests/minute: {stats['max_requests_per_minute']}")
    print(f"   Is rate limited: {stats['is_rate_limited']}")
    print(f"   Consecutive 429s: {stats['consecutive_429s']}")
    print(f"   Backoff multiplier: {stats['backoff_multiplier']:.2f}")


def test_basic_rate_limiting():
    """Test basic rate limiting with credit-only tier limits."""
    print("\n" + "="*60)
    print("TEST 1: Basic Rate Limiting (Credit-Only Tier)")
    print("="*60)
    print("Configuration:")
    print("  - Max: 5 requests/minute")
    print("  - Burst: 3 requests")
    print("  - Min interval: 1.2 seconds")
    print()
    
    limiter = get_rate_limiter()
    
    # Test burst capacity (should get 3 requests quickly)
    print("Testing burst capacity (should get 3 requests)...")
    for i in range(3):
        start = time.time()
        success = limiter.acquire(timeout=5.0)
        elapsed = time.time() - start
        
        if success:
            print(f"✅ Request {i+1}: Acquired in {elapsed:.3f}s")
        else:
            print(f"❌ Request {i+1}: TIMEOUT after {elapsed:.3f}s")
    
    print_stats()
    
    # Test rate limiting (4th request should be delayed)
    print("\nTesting rate limiting (4th request should wait ~1.2s)...")
    start = time.time()
    success = limiter.acquire(timeout=5.0)
    elapsed = time.time() - start
    
    if success:
        print(f"✅ Request 4: Acquired in {elapsed:.3f}s (expected: ~1.2s)")
    else:
        print(f"❌ Request 4: TIMEOUT after {elapsed:.3f}s")
    
    print_stats()
    
    # Test sustained rate (5th request should also wait)
    print("\nTesting sustained rate (5th request should wait ~1.2s)...")
    start = time.time()
    success = limiter.acquire(timeout=5.0)
    elapsed = time.time() - start
    
    if success:
        print(f"✅ Request 5: Acquired in {elapsed:.3f}s (expected: ~1.2s)")
    else:
        print(f"❌ Request 5: TIMEOUT after {elapsed:.3f}s")
    
    print_stats()
    
    print("\n✅ Test 1 Complete!")


def test_minute_limit():
    """Test that we don't exceed 5 requests per minute."""
    print("\n" + "="*60)
    print("TEST 2: Minute Limit (Max 5 requests/minute)")
    print("="*60)
    print("Attempting 6 requests in quick succession...")
    print("Expected: First 3 quick, then 1.2s delays, 6th should timeout")
    print()
    
    limiter = get_rate_limiter()
    
    start_time = time.time()
    successful_requests = 0
    
    for i in range(6):
        request_start = time.time()
        success = limiter.acquire(timeout=2.0)  # Short timeout for 6th request
        elapsed = time.time() - request_start
        total_elapsed = time.time() - start_time
        
        if success:
            successful_requests += 1
            print(f"✅ Request {i+1}: Acquired in {elapsed:.3f}s (total: {total_elapsed:.3f}s)")
        else:
            print(f"❌ Request {i+1}: TIMEOUT after {elapsed:.3f}s (total: {total_elapsed:.3f}s)")
    
    print(f"\n📊 Results:")
    print(f"   Successful requests: {successful_requests}/6")
    print(f"   Total time: {time.time() - start_time:.3f}s")
    print(f"   Expected: 5 successful (6th should timeout)")
    
    print_stats()
    
    if successful_requests == 5:
        print("\n✅ Test 2 PASSED: Correctly limited to 5 requests")
    else:
        print(f"\n⚠️  Test 2 WARNING: Got {successful_requests} requests (expected 5)")


def test_429_backoff():
    """Test exponential backoff on 429 errors."""
    print("\n" + "="*60)
    print("TEST 3: Exponential Backoff on 429 Errors")
    print("="*60)
    print("Simulating 429 error and testing backoff...")
    print()
    
    limiter = get_rate_limiter()
    
    # Simulate a 429 error
    print("Simulating 429 error...")
    limiter.report_429()
    
    print_stats()
    
    # Try to acquire token (should be blocked)
    print("\nAttempting to acquire token (should be blocked for 30s)...")
    start = time.time()
    success = limiter.acquire(timeout=2.0)  # Short timeout to avoid waiting 30s
    elapsed = time.time() - start
    
    if success:
        print(f"⚠️  Request succeeded in {elapsed:.3f}s (unexpected!)")
    else:
        print(f"✅ Request blocked for {elapsed:.3f}s (expected: blocked)")
    
    print_stats()
    
    print("\n✅ Test 3 Complete!")
    print("Note: Full 30s backoff not tested (would take too long)")


def test_timing_accuracy():
    """Test that minimum interval is enforced accurately."""
    print("\n" + "="*60)
    print("TEST 4: Timing Accuracy (1.2s minimum interval)")
    print("="*60)
    print("Testing 5 consecutive requests with timing...")
    print()
    
    limiter = get_rate_limiter()
    
    intervals = []
    last_time = None
    
    for i in range(5):
        start = time.time()
        success = limiter.acquire(timeout=5.0)
        
        if success:
            current_time = time.time()
            if last_time is not None:
                interval = current_time - last_time
                intervals.append(interval)
                print(f"✅ Request {i+1}: Interval = {interval:.3f}s")
            else:
                print(f"✅ Request {i+1}: First request (no interval)")
            last_time = current_time
        else:
            print(f"❌ Request {i+1}: TIMEOUT")
    
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        print(f"\n📊 Interval Statistics:")
        print(f"   Average: {avg_interval:.3f}s")
        print(f"   Minimum: {min_interval:.3f}s")
        print(f"   Maximum: {max_interval:.3f}s")
        print(f"   Expected: ~1.2s minimum")
        
        if min_interval >= 1.2:
            print(f"\n✅ Test 4 PASSED: All intervals >= 1.2s")
        else:
            print(f"\n⚠️  Test 4 WARNING: Minimum interval {min_interval:.3f}s < 1.2s")
    
    print_stats()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REPLICATE RATE LIMITER TEST SUITE")
    print("Credit-Only Tier Configuration")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test 1: Basic rate limiting
        test_basic_rate_limiting()
        
        # Wait a bit before next test
        print("\n⏳ Waiting 10 seconds before next test...")
        time.sleep(10)
        
        # Test 2: Minute limit
        test_minute_limit()
        
        # Wait a bit before next test
        print("\n⏳ Waiting 10 seconds before next test...")
        time.sleep(10)
        
        # Test 3: 429 backoff
        test_429_backoff()
        
        # Test 4: Timing accuracy
        print("\n⏳ Waiting 10 seconds before next test...")
        time.sleep(10)
        test_timing_accuracy()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETE")
        print("="*60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
