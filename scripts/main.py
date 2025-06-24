#!/usr/bin/env python3

import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Configuration
DATABASE_FILE = '../database/api-collections.json'
MODEL_NAME = "openai/clip-vit-large-patch14"
API_KEYS = {"demo_key_12345": {"name": "Demo User", "tier": "free"}}  # Simple auth for now

app = FastAPI(
    title="Visual Search API",
    description="Add visual similarity search to your applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for model
model = None
processor = None
database = None

# Pydantic models for API
class Collection(BaseModel):
    name: str
    description: Optional[str] = ""

class ImageItem(BaseModel):
    id: str
    filename: str
    upload_date: str
    collection_id: str

class SearchResult(BaseModel):
    image_id: str
    filename: str
    similarity: float
    collection_id: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_searched: int
    query_time_ms: float

class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    image_count: int
    created_date: str

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return API_KEYS[api_key]

# Database functions
def load_database():
    """Load the collections database"""
    global database
    try:
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                database = json.load(f)
        else:
            database = {"collections": {}, "images": {}}
        print(f"✅ Database loaded with {len(database['collections'])} collections")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        database = {"collections": {}, "images": {}}

def save_database():
    """Save database to file"""
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f, indent=2)

def load_model():
    """Load the CLIP model once at startup"""
    global model, processor
    print("🔄 Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("✅ Model loaded!")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))

def get_image_embedding(image):
    """Get CLIP embedding for an image"""
    global model, processor
    
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
    
    return embedding.tolist()

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup"""
    load_model()
    load_database()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Visual Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "collections": len(database['collections']) if database else 0,
        "total_images": len(database['images']) if database else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_loaded": database is not None,
        "collections": len(database['collections']) if database else 0,
        "images": len(database['images']) if database else 0
    }

@app.post("/collections", response_model=CollectionResponse)
async def create_collection(
    collection: Collection,
    user: dict = Depends(verify_api_key)
):
    """Create a new image collection"""
    collection_id = str(uuid.uuid4())
    
    database['collections'][collection_id] = {
        "id": collection_id,
        "name": collection.name,
        "description": collection.description,
        "created_date": datetime.now().isoformat(),
        "user": user["name"],
        "image_count": 0
    }
    
    save_database()
    
    return CollectionResponse(**database['collections'][collection_id])

@app.get("/collections", response_model=List[CollectionResponse])
async def list_collections(user: dict = Depends(verify_api_key)):
    """List all collections for the authenticated user"""
    user_collections = []
    for collection in database['collections'].values():
        if collection.get("user") == user["name"]:
            user_collections.append(CollectionResponse(**collection))
    
    return user_collections

@app.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: str,
    user: dict = Depends(verify_api_key)
):
    """Delete a collection and all its images"""
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Remove all images in this collection
    images_to_remove = [img_id for img_id, img in database['images'].items() 
                       if img.get('collection_id') == collection_id]
    
    for img_id in images_to_remove:
        del database['images'][img_id]
    
    # Remove collection
    del database['collections'][collection_id]
    save_database()
    
    return {"message": f"Collection {collection_id} deleted successfully"}

@app.post("/collections/{collection_id}/images")
async def add_image_to_collection(
    collection_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(verify_api_key)
):
    """Add an image to a collection"""
    # Verify collection exists and user has access
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        embedding = get_image_embedding(image)
        
        # Create image record
        image_id = str(uuid.uuid4())
        image_record = {
            "id": image_id,
            "filename": file.filename,
            "collection_id": collection_id,
            "embedding": embedding,
            "upload_date": datetime.now().isoformat(),
            "user": user["name"]
        }
        
        database['images'][image_id] = image_record
        
        # Update collection image count
        database['collections'][collection_id]['image_count'] += 1
        
        save_database()
        
        return {
            "message": "Image added successfully",
            "image_id": image_id,
            "filename": file.filename,
            "collection_id": collection_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/collections/{collection_id}/images", response_model=List[ImageItem])
async def list_images_in_collection(
    collection_id: str,
    user: dict = Depends(verify_api_key)
):
    """List all images in a collection"""
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    collection_images = []
    for image in database['images'].values():
        if image.get('collection_id') == collection_id:
            collection_images.append(ImageItem(
                id=image['id'],
                filename=image['filename'],
                upload_date=image['upload_date'],
                collection_id=image['collection_id']
            ))
    
    return collection_images

@app.post("/collections/{collection_id}/search", response_model=SearchResponse)
async def search_similar_images(
    collection_id: str,
    file: UploadFile = File(...),
    top_k: int = 5,
    user: dict = Depends(verify_api_key)
):
    """Search for similar images in a collection"""
    import time
    start_time = time.time()
    
    # Verify collection access
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process query image
        query_image = Image.open(file.file)
        query_embedding = get_image_embedding(query_image)
        
        # Find similar images in the collection
        similarities = []
        collection_images = [img for img in database['images'].values() 
                           if img.get('collection_id') == collection_id]
        
        for image_data in collection_images:
            similarity = cosine_similarity(query_embedding, image_data['embedding'])
            similarities.append({
                'image_id': image_data['id'],
                'filename': image_data['filename'],
                'similarity': similarity,
                'collection_id': collection_id
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        results = similarities[:top_k]
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            results=[SearchResult(**result) for result in results],
            total_searched=len(collection_images),
            query_time_ms=round(query_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching images: {str(e)}")

@app.get("/usage")
async def get_usage_stats(user: dict = Depends(verify_api_key)):
    """Get API usage statistics for the user"""
    user_collections = len([c for c in database['collections'].values() 
                           if c.get("user") == user["name"]])
    user_images = len([img for img in database['images'].values() 
                      if img.get("user") == user["name"]])
    
    return {
        "user": user["name"],
        "tier": user["tier"],
        "collections": user_collections,
        "total_images": user_images,
        "api_calls_today": 0,  # TODO: Implement usage tracking
        "api_calls_month": 0
    }

if __name__ == "__main__":
    print("🚀 Starting Visual Search API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔑 Demo API Key: demo_key_12345")
    print("📖 Interactive docs: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)