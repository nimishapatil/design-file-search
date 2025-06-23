#!/usr/bin/env python3

import os
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration
DATABASE_FILE = '../database/local-embeddings.json'

def load_database():
    """Load the embeddings database"""
    try:
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        print(f"💡 Make sure you've run local-scan-images.py first")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_image_embedding(image_path, model, processor):
    """Get CLIP embedding for a query image"""
    try:
        # Load and process image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image for CLIP
        inputs = processor(images=image, return_tensors="pt")
        
        # Get image features (embeddings)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embedding = image_features.squeeze().numpy()
        
        return embedding
        
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def search_similar_images(query_image_path, top_k=5):
    """Find similar images to the query image"""
    print(f"🔍 Searching for images similar to: {query_image_path}")
    
    # Load database
    db = load_database()
    if not db or not db['images']:
        print("❌ No images found in database")
        return
    
    print(f"📚 Searching through {len(db['images'])} images...")
    
    # Load CLIP model
    print("📥 Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Get embedding for query image
    print("🔄 Processing query image...")
    try:
        query_embedding = get_image_embedding(query_image_path, model, processor)
    except Exception as e:
        print(f"❌ Error processing query image: {e}")
        return
    
    # Calculate similarities
    print("📊 Calculating similarities...")
    similarities = []
    
    for img_data in db['images']:
        db_embedding = np.array(img_data['embedding'])
        similarity = cosine_similarity(query_embedding, db_embedding)
        
        similarities.append({
            'filename': img_data['filename'],
            'path': img_data['path'],
            'similarity': similarity
        })
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Show results
    print(f"\n🎯 Top {min(top_k, len(similarities))} most similar images:")
    print("=" * 80)
    
    for i, result in enumerate(similarities[:top_k]):
        similarity_percent = result['similarity'] * 100
        
        # Visual similarity indicator
        if similarity_percent > 90:
            indicator = "🟢 Excellent match"
        elif similarity_percent > 75:
            indicator = "🟡 Good match"
        elif similarity_percent > 50:
            indicator = "🟠 Moderate match"
        else:
            indicator = "🔴 Weak match"
        
        print(f"{i+1}. {result['filename']}")
        print(f"   Similarity: {similarity_percent:.1f}% {indicator}")
        print(f"   Path: {result['path']}")
        print()
    
    return similarities[:top_k]

def show_all_similarities():
    """Show similarity matrix for all images"""
    db = load_database()
    if not db or len(db['images']) < 2:
        print("❌ Need at least 2 images to compare")
        return
    
    print(f"🔗 Similarity matrix for all {len(db['images'])} images:")
    print("=" * 60)
    
    images = db['images']
    
    # Show top 5 most similar pairs
    similarities = []
    
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            emb1 = np.array(images[i]['embedding'])
            emb2 = np.array(images[j]['embedding'])
            similarity = cosine_similarity(emb1, emb2)
            
            similarities.append({
                'image1': images[i]['filename'],
                'image2': images[j]['filename'],
                'similarity': similarity
            })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("🏆 Top 10 most similar image pairs:")
    for i, sim in enumerate(similarities[:10]):
        similarity_percent = sim['similarity'] * 100
        print(f"{i+1:2d}. {sim['image1'][:30]:<30} ↔ {sim['image2'][:30]:<30} ({similarity_percent:.1f}%)")

def interactive_search():
    """Interactive search mode"""
    db = load_database()
    if not db or not db['images']:
        print("❌ No images found in database")
        return
    
    print("🔍 Interactive Image Search")
    print("=" * 40)
    print(f"Database contains {len(db['images'])} images")
    print("\nCommands:")
    print("  search <image_path>  - Find similar images")
    print("  list                 - Show all images in database")
    print("  matrix               - Show similarity matrix")
    print("  quit                 - Exit")
    print()
    
    while True:
        try:
            command = input("🔍 > ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print("👋 Goodbye!")
                break
            elif command == 'list':
                print("\n📚 Images in database:")
                for i, img in enumerate(db['images'], 1):
                    print(f"{i:2d}. {img['filename']}")
                print()
            elif command == 'matrix':
                show_all_similarities()
                print()
            elif command.startswith('search '):
                image_path = command[7:].strip()
                if os.path.exists(image_path):
                    search_similar_images(image_path)
                else:
                    print(f"❌ Image not found: {image_path}")
                print()
            else:
                print("❓ Unknown command. Try 'search <image_path>', 'list', 'matrix', or 'quit'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Search for similar images using local CLIP')
    parser.add_argument('--query', '-q', type=str, help='Path to query image')
    parser.add_argument('--top', '-t', type=int, default=5, help='Number of results to show (default: 5)')
    parser.add_argument('--matrix', '-m', action='store_true', help='Show similarity matrix for all images')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive search mode')
    
    args = parser.parse_args()
    
    if args.matrix:
        show_all_similarities()
    elif args.query:
        search_similar_images(args.query, args.top)
    elif args.interactive:
        interactive_search()
    else:
        # Default to interactive mode
        interactive_search()

if __name__ == "__main__":
    main()