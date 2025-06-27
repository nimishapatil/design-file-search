#!/usr/bin/env python3
"""
Basic example of using design_search library
"""

import sys
import os

# Add the parent directory to the path so we can import design_search
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from design_search import create_collection, search_images, search_text

def main():
    print("🚀 Design Search Library - Basic Example")
    print("=" * 50)
    
    # Create a collection
    print("Creating collection...")
    collection = create_collection("demo_collection", model="clip-vit-base")
    print(f"✅ Created: {collection}")
    
    # Add some images (using test images from load-testing)
    test_images = [
        "../../load-testing/load-test-data/Biz Marketing Logo Horizontal.png",
        "../../load-testing/load-test-data/Retro Boombox.jpeg",
        "../../load-testing/load-test-data/Alpine Serenity at Dawn.jpeg",
        "../../load-testing/load-test-data/Profile Picture Nim (1).png"
    ]
    
    print("\nAdding images to collection...")
    added_images = []
    for image_path in test_images:
        if os.path.exists(image_path):
            try:
                image_id = collection.add_image(image_path)
                added_images.append(image_path)
                print(f"✅ Added: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Failed to add {os.path.basename(image_path)}: {e}")
    
    if not added_images:
        print("❌ No images were added. Make sure test images exist.")
        return
    
    # Show collection stats
    stats = collection.get_stats()
    print(f"\n📊 Collection Stats:")
    print(f"   Name: {stats['name']}")
    print(f"   Images: {stats['image_count']}")
    print(f"   Model: {stats['model']['name']}")
    
    # Test image search
    print(f"\n🔍 Image Search Test:")
    if added_images:
        query_image = added_images[0]  # Use first added image as query
        print(f"Searching for images similar to: {os.path.basename(query_image)}")
        
        results = search_images(collection, query_image, top_k=3)
        for i, result in enumerate(results, 1):
            similarity_pct = result['similarity'] * 100
            print(f"  {i}. {result['filename']}: {similarity_pct:.1f}% similarity")
    
    # Test text search
    print(f"\n💬 Text Search Test:")
    test_queries = ["logo", "mountain", "person", "audio"]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = search_text(collection, query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                similarity_pct = result['similarity'] * 100
                print(f"  {i}. {result['filename']}: {similarity_pct:.1f}% match")
        else:
            print("  No results found")
    
    print(f"\n🎉 Demo completed!")
    print(f"Collection contains {len(collection.images)} images")
    print(f"Try running: python -c \"from design_search import create_collection; print('Import works!')\"")

if __name__ == "__main__":
    main()