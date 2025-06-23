#!/usr/bin/env python3

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration
IMAGES_FOLDER = '../test-images'
DATABASE_FILE = '../database/local-embeddings.json'
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

def load_database():
    """Load existing database or create new one"""
    try:
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Creating new database... ({e})")
    return {"images": []}

def save_database(db):
    """Save database to file"""
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    
    with open(DATABASE_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_image_embedding(image_path, model, processor):
    """Get CLIP embedding for an image"""
    try:
        # Load and process image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary (handles PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image for CLIP
        inputs = processor(images=image, return_tensors="pt")
        
        # Get image features (embeddings)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
        # Normalize the features (this is important for similarity search)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embedding = image_features.squeeze().numpy()
        
        return embedding.tolist()  # Convert to list for JSON serialization
        
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def scan_images():
    """Main function to scan images and create embeddings"""
    print("🔍 Starting local CLIP image scan...")
    
    # Load the CLIP model (this will download ~600MB the first time)
    print("📥 Loading CLIP model (this may take a moment the first time)...")
    model_name = "openai/clip-vit-base-patch32"  # Smaller, faster model
    
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have internet connection for the initial download")
        return
    
    # Load existing database
    db = load_database()
    print(f"📚 Loaded database with {len(db['images'])} existing images")
    
    # Get list of image files
    images_path = Path(__file__).parent / IMAGES_FOLDER
    if not images_path.exists():
        print(f"❌ Images folder not found: {images_path}")
        return
    
    # Find all image files
    image_files = []
    for file_path in images_path.iterdir():
        if file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file_path)
    
    print(f"📸 Found {len(image_files)} image files")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_path in image_files:
        filename = image_path.name
        
        # Check if already processed
        existing_image = next((img for img in db['images'] if img['filename'] == filename), None)
        if existing_image:
            print(f"⏭️  Skipping {filename} (already processed)")
            skipped += 1
            continue
        
        try:
            print(f"🔄 Processing {filename}...")
            
            # Get embedding
            embedding = get_image_embedding(image_path, model, processor)
            
            # Add to database
            db['images'].append({
                'filename': filename,
                'path': str(image_path.absolute()),
                'embedding': embedding,
                'embedding_model': model_name,
                'processedAt': datetime.now().isoformat()
            })
            
            # Save database after each image
            save_database(db)
            
            processed += 1
            print(f"✅ Processed {filename} ({processed}/{len(image_files)})")
            print(f"   Embedding dimensions: {len(embedding)}")
            
        except Exception as error:
            print(f"❌ Error processing {filename}: {error}")
            errors += 1
    
    print("\n🎉 Local scan complete!")
    print(f"✅ Processed: {processed} images")
    print(f"⏭️  Skipped: {skipped} images")
    print(f"❌ Errors: {errors} images")
    print(f"💾 Database saved to: {DATABASE_FILE}")
    
    if processed > 0:
        print("\n🚀 Ready to build local search functionality!")
        print("📊 Model used: openai/clip-vit-base-patch32")

def test_similarity():
    """Test function to show similarity between processed images"""
    db = load_database()
    images = db['images']
    
    if len(images) < 2:
        print("Need at least 2 processed images to test similarity")
        return
    
    print(f"\n🧪 Testing similarity between first 2 images:")
    
    img1 = images[0]
    img2 = images[1]
    
    similarity = cosine_similarity(np.array(img1['embedding']), np.array(img2['embedding']))
    
    print(f"📸 {img1['filename']}")
    print(f"📸 {img2['filename']}")
    print(f"🔗 Similarity score: {similarity:.3f}")
    print(f"   (1.0 = identical, 0.0 = completely different)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_similarity()
    else:
        scan_images()