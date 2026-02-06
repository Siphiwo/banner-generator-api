import requests
import json
import time

# Test the API
print("Testing Banner Builder API...")
print("=" * 60)

# 1. Health check
print("\n1. Testing health endpoint...")
response = requests.get("http://127.0.0.1:8000/health")
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# 2. Create a new job
print("\n2. Creating a new job...")
with open("storage/jobs/34ccb744-7022-4bbf-bfd1-2cb7566a8947/master.webp", "rb") as f:
    files = {"master_banner": f}
    data = {
        "outputs": json.dumps([
            {"width": 300, "height": 250},
            {"width": 728, "height": 90},
            {"width": 160, "height": 600},
            {"width": 1200, "height": 628}
        ])
    }
    response = requests.post("http://127.0.0.1:8000/api/v1/jobs", files=files, data=data)

print(f"   Status: {response.status_code}")
if response.status_code == 200:
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"   Job ID: {job_id}")
    print(f"   Status: {job_data['status']}")
    print(f"   Outputs requested: {len(job_data['outputs'])}")
    
    # 3. Wait a moment for processing
    print("\n3. Waiting for job to process...")
    time.sleep(2)
    
    # 4. Get job details
    print("\n4. Fetching job details...")
    response = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
    if response.status_code == 200:
        job_details = response.json()
        print(f"   Status: {job_details['status']}")
        print(f"   Created: {job_details['created_at']}")
        
        if job_details.get('outputs'):
            print(f"\n   Generated Outputs:")
            for output in job_details['outputs']:
                print(f"   - {output['width']}x{output['height']}: {output.get('url', 'N/A')}")
        
        # Show quality checks if available
        if job_details.get('quality_checks'):
            print(f"\n   Quality Checks:")
            for qc in job_details['quality_checks']:
                print(f"   - {qc['output_size']}: Score {qc['quality_score']:.2f}, " +
                      f"Confidence {qc['confidence']:.2f}, " +
                      f"Manual Review: {qc['needs_manual_review']}")
                if qc.get('warnings'):
                    for warning in qc['warnings']:
                        print(f"     ⚠️  {warning}")
        
        # Show layout plans if available
        if job_details.get('layout_plans'):
            print(f"\n   Layout Strategies:")
            for plan in job_details['layout_plans']:
                print(f"   - {plan['output_size']}: {plan.get('strategy_class', 'N/A')}")
                print(f"     Scaling: {plan.get('scaling_mode', 'N/A')}")
    else:
        print(f"   Error: {response.status_code}")
        print(f"   {response.text}")
    
    # 5. List all jobs
    print("\n5. Listing all jobs...")
    response = requests.get("http://127.0.0.1:8000/api/v1/jobs")
    if response.status_code == 200:
        jobs = response.json()
        print(f"   Total jobs: {len(jobs)}")
        for job in jobs[:3]:  # Show first 3
            print(f"   - {job['job_id']}: {job['status']}")
else:
    print(f"   Error: {response.status_code}")
    print(f"   {response.text}")

print("\n" + "=" * 60)
print("Test complete!")
