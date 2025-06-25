#!/usr/bin/env python3
import requests
import time
import threading
import json
import os

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer demo_key_12345"}

def create_collection():
    """Create a test collection"""
    response = requests.post(f"{BASE_URL}/collections", 
        json={
            "name": f"Load Test {time.time()}",
            "description": "Testing collection",
            "model": {"type": "builtin", "name": "clip-vit-base"}
        },
        headers=HEADERS
    )
    if response.status_code == 200:
        return response.json()["id"]
    else:
        print(f"Failed to create collection: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_basic_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing basic endpoints...")
    
    endpoints = [
        ("Health Check", "GET", "/health"),
        ("List Models", "GET", "/models"),
        ("List Collections", "GET", "/collections"),
    ]
    
    for name, method, endpoint in endpoints:
        try:
            start = time.time()
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS)
            end = time.time()
            
            response_time = (end - start) * 1000
            
            if response.status_code == 200:
                print(f"  ✅ {name}: {response_time:.1f}ms")
            else:
                print(f"  ❌ {name}: {response.status_code} ({response_time:.1f}ms)")
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")

def test_single_user_flow():
    """Test complete user flow with one user"""
    print("\n🧪 Testing single user flow...")
    
    # 1. Create collection
    print("  Creating collection...")
    collection_id = create_collection()
    if not collection_id:
        print("  ❌ Failed to create collection")
        return False
    print(f"  ✅ Created collection: {collection_id}")
    
    # 2. Get available test images
    test_images = [f for f in os.listdir("load-test-data") if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not test_images:
        print("  ❌ No test images found in load-test-data/")
        return False
    
    test_image = test_images[0]
    print(f"  Using test image: {test_image}")
    
    # 3. Upload image
    print("  Uploading image...")
    try:
        with open(f"load-test-data/{test_image}", "rb") as f:
            files = {"file": (test_image, f, "image/jpeg")}
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/collections/{collection_id}/images",
                files=files,
                headers=HEADERS
            )
            end = time.time()
        
        upload_time = (end - start) * 1000
        
        if response.status_code == 200:
            print(f"  ✅ Image uploaded: {upload_time:.1f}ms")
        else:
            print(f"  ❌ Upload failed: {response.status_code} ({upload_time:.1f}ms)")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Upload error: {e}")
        return False
    
    # 4. Search for similar images
    print("  Searching for similar images...")
    try:
        with open(f"load-test-data/{test_image}", "rb") as f:
            files = {"file": (test_image, f, "image/jpeg")}
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/collections/{collection_id}/search",
                files=files,
                headers=HEADERS
            )
            end = time.time()
        
        search_time = (end - start) * 1000
        
        if response.status_code == 200:
            results = response.json()
            num_results = len(results.get('results', []))
            print(f"  ✅ Search completed: {search_time:.1f}ms, {num_results} results")
        else:
            print(f"  ❌ Search failed: {response.status_code} ({search_time:.1f}ms)")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Search error: {e}")
        return False
    
    return True

def test_concurrent_users(num_users=3):
    """Test multiple concurrent users"""
    print(f"\n🧪 Testing {num_users} concurrent users...")
    
    def user_simulation(user_id):
        try:
            print(f"  User {user_id}: Starting...")
            
            # Create collection
            collection_id = create_collection()
            if not collection_id:
                print(f"  User {user_id}: ❌ Failed to create collection")
                return False
            
            # Get test image
            test_images = [f for f in os.listdir("load-test-data") if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not test_images:
                print(f"  User {user_id}: ❌ No test images")
                return False
            
            test_image = test_images[user_id % len(test_images)]  # Use different images
            
            # Upload image
            with open(f"load-test-data/{test_image}", "rb") as f:
                files = {"file": (test_image, f, "image/jpeg")}
                response = requests.post(
                    f"{BASE_URL}/collections/{collection_id}/images",
                    files=files,
                    headers=HEADERS
                )
            
            if response.status_code != 200:
                print(f"  User {user_id}: ❌ Upload failed ({response.status_code})")
                return False
            
            # Search
            with open(f"load-test-data/{test_image}", "rb") as f:
                files = {"file": (test_image, f, "image/jpeg")}
                response = requests.post(
                    f"{BASE_URL}/collections/{collection_id}/search",
                    files=files,
                    headers=HEADERS
                )
            
            if response.status_code == 200:
                results = response.json()
                num_results = len(results.get('results', []))
                print(f"  User {user_id}: ✅ Complete! {num_results} search results")
                return True
            else:
                print(f"  User {user_id}: ❌ Search failed ({response.status_code})")
                return False
                
        except Exception as e:
            print(f"  User {user_id}: ❌ Error - {e}")
            return False
    
    # Start threads
    threads = []
    results = []
    
    def thread_wrapper(user_id):
        result = user_simulation(user_id)
        results.append(result)
    
    for i in range(num_users):
        thread = threading.Thread(target=thread_wrapper, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    successful = sum(results)
    print(f"  📊 Results: {successful}/{num_users} users successful")
    
    return successful == num_users

def main():
    print("�� Starting Simple Load Test")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API health check failed!")
            print("Make sure your API is running: python main-byom.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API!")
        print("Make sure your API is running on localhost:8000")
        print("Run: cd ../scripts && python main-byom.py")
        return
    
    print("✅ API is running!")
    
    # Run tests
    test_basic_endpoints()
    
    success = test_single_user_flow()
    if not success:
        print("\n❌ Single user flow failed! Fix this before testing concurrent users.")
        return
    
    # Test with 3 concurrent users first
    success = test_concurrent_users(3)
    if success:
        print("\n✅ 3 concurrent users successful! Trying 5...")
        test_concurrent_users(5)
    else:
        print("\n❌ Concurrent users failed. Your API may need optimization.")
    
    print("\n🎉 Load test complete!")
    print("\nNext steps:")
    print("1. If tests passed: You're ready for demos!")
    print("2. If tests failed: Check the error messages above")
    print("3. Monitor system resources during tests")

if __name__ == "__main__":
    main()
