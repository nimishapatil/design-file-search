#!/usr/bin/env python3

import os
import json
import uuid
import requests
import base64
from datetime import datetime
from typing import List, Optional, Dict, Any
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import uvicorn

# Configuration
DATABASE_FILE = os.path.join(os.path.dirname(__file__), '..', 'database', 'byom-collections.json')
API_KEYS = {"demo_key_12345": {"name": "Demo User", "tier": "free"}}

app = FastAPI(
    title="Visual Search API - Multi-Model",
    description="Add visual similarity search to your applications with any AI model",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Global variables
built_in_models = {}
database = None

# Enhanced Pydantic models
class ModelConfig(BaseModel):
    type: str  # "builtin", "endpoint", "openai"
    name: Optional[str] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None
    config: Optional[Dict[str, Any]] = {}
    
    @validator('type')
    def validate_model_type(cls, v):
        if v not in ['builtin', 'endpoint', 'openai']:
            raise ValueError('Model type must be: builtin, endpoint, or openai')
        return v

class Collection(BaseModel):
    name: str
    description: Optional[str] = ""
    model: ModelConfig
class ModelInfo(BaseModel):
    name: str
    type: str
    dimensions: int
    description: str
    pricing: Optional[str] = None

class SearchResult(BaseModel):
    image_id: str
    filename: str
    similarity: float
    collection_id: str

class TextSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class TextSearchResponse(BaseModel):
    results: List[SearchResult]
    total_searched: int
    query_time_ms: float
    model_used: str
    query_text: str
class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_searched: int
    query_time_ms: float
    model_used: str

class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    image_count: int
    created_date: str
    model: ModelConfig

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return API_KEYS[api_key]

# Model Management
class ModelManager:
    def __init__(self):
        self.built_in_models = {}
        
    def load_builtin_model(self, model_name: str):
        """Load a built-in CLIP model"""
        if model_name not in self.built_in_models:
            print(f"🔄 Loading built-in model: {model_name}")
            if model_name == "clip-vit-large":
                model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.built_in_models[model_name] = {"model": model, "processor": processor, "dimensions": 768}
            elif model_name == "clip-vit-base":
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.built_in_models[model_name] = {"model": model, "processor": processor, "dimensions": 512}
            else:
                raise ValueError(f"Unknown built-in model: {model_name}")
            print(f"✅ Model {model_name} loaded!")
        
        return self.built_in_models[model_name]
    
    def get_builtin_embedding(self, model_name: str, image: Image.Image) -> List[float]:
        """Get embedding from built-in model"""
        model_data = self.load_builtin_model(model_name)
        model = model_data["model"]
        processor = model_data["processor"]
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.squeeze().numpy().tolist()
    
    def get_endpoint_embedding(self, endpoint: str, api_key: str, image: Image.Image) -> List[float]:
        """Get embedding from external endpoint"""
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Call external endpoint
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "image": image_b64,
            "format": "jpeg"
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Expected response format: {"embedding": [0.1, 0.2, ...]}
            if "embedding" not in result:
                raise ValueError("External model must return {'embedding': [...]}")
            
            return result["embedding"]
        
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"External model error: {str(e)}")
    
    def get_openai_embedding(self, api_key: str, image: Image.Image) -> List[float]:
        """Get embedding from OpenAI (placeholder for future)"""
        # This would integrate with OpenAI's vision API when available
        raise HTTPException(status_code=501, detail="OpenAI integration coming soon")
    
    def get_embedding(self, model_config: ModelConfig, image: Image.Image) -> List[float]:
        """Get embedding using specified model configuration"""
        if model_config.type == "builtin":
            return self.get_builtin_embedding(model_config.name, image)
        elif model_config.type == "endpoint":
            return self.get_endpoint_embedding(model_config.endpoint, model_config.api_key, image)
        elif model_config.type == "openai":
            return self.get_openai_embedding(model_config.api_key, image)
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")
        
    def get_text_embedding(self, model_config: ModelConfig, text: str) -> List[float]:
        """Get embedding for text using specified model configuration"""
        if model_config.type == "builtin":
            return self.get_builtin_text_embedding(model_config.name, text)
        elif model_config.type == "endpoint":
            return self.get_endpoint_text_embedding(model_config.endpoint, model_config.api_key, text)
        elif model_config.type == "openai":
            return self.get_openai_text_embedding(model_config.api_key, text)
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")

    def get_builtin_text_embedding(self, model_name: str, text: str) -> List[float]:
        """Get text embedding from built-in CLIP model"""
        model_data = self.load_builtin_model(model_name)
        model = model_data["model"]
        processor = model_data["processor"]

        # Process text for CLIP
        inputs = processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze().numpy().tolist()

    def get_endpoint_text_embedding(self, endpoint: str, api_key: str, text: str) -> List[float]:
        """Get text embedding from external endpoint"""
        # Call external endpoint for text
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "text": text,
            "type": "text"
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            # Expected response format: {"embedding": [0.1, 0.2, ...]}
            if "embedding" not in result:
                raise ValueError("External model must return {'embedding': [...]}" )

            return result["embedding"]

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"External model error: {str(e)}")

    def get_openai_text_embedding(self, api_key: str, text: str) -> List[float]:
        """Get text embedding from OpenAI (placeholder for future)"""
        # This would integrate with OpenAI's text embedding API when available
        raise HTTPException(status_code=501, detail="OpenAI text integration coming soon")

# Initialize model manager
model_manager = ModelManager()

# Database functions
def load_database():
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
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f, indent=2)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))

# API Routes
@app.on_event("startup")
async def startup_event():
    load_database()

@app.get("/")
async def root():
    return {
        "message": "Visual Search API - Multi-Model",
        "version": "2.0.0",
        "features": ["builtin_models", "custom_endpoints", "model_marketplace"],
        "docs": "/docs",
        "collections": len(database['collections']) if database else 0,
        "total_images": len(database['images']) if database else 0
    }

@app.get("/models")
async def list_available_models():
    """List all available models"""
    return [
        {
            "name": "clip-vit-large",
            "type": "builtin",
            "dimensions": 768,
            "description": "OpenAI CLIP ViT-Large - High accuracy, slower processing",
            "pricing": "Free"
        },
        {
            "name": "clip-vit-base", 
            "type": "builtin",
            "dimensions": 512,
            "description": "OpenAI CLIP ViT-Base - Good accuracy, faster processing", 
            "pricing": "Free"
        },
        {
            "name": "custom-endpoint",
            "type": "endpoint",
            "dimensions": 0,
            "description": "Bring your own model via HTTP endpoint",
            "pricing": "Your infrastructure costs"
        }
    ]

@app.post("/collections", response_model=CollectionResponse)
async def create_collection(
    collection: Collection,
    user: dict = Depends(verify_api_key)
):
    """Create a new image collection with specified model"""
    collection_id = str(uuid.uuid4())
    
    # Validate model configuration
    if collection.model.type == "builtin":
        if collection.model.name not in ["clip-vit-large", "clip-vit-base"]:
            raise HTTPException(status_code=400, detail="Invalid built-in model name")
    elif collection.model.type == "endpoint":
        if not collection.model.endpoint:
            raise HTTPException(status_code=400, detail="Endpoint URL required for endpoint models")
    
    # Get model dimensions
    if collection.model.type == "builtin":
        model_data = model_manager.load_builtin_model(collection.model.name)
        dimensions = model_data["dimensions"]
    else:
        dimensions = collection.model.dimensions or 512  # Default
    
    database['collections'][collection_id] = {
        "id": collection_id,
        "name": collection.name,
        "description": collection.description,
        "created_date": datetime.now().isoformat(),
        "user": user["name"],
        "image_count": 0,
        "model": collection.model.dict(),
        "dimensions": dimensions
    }
    
    save_database()
    
    return CollectionResponse(
        id=collection_id,
        name=collection.name,
        description=collection.description,
        image_count=0,
        created_date=database['collections'][collection_id]["created_date"],
        model=collection.model
    )

@app.get("/collections", response_model=List[CollectionResponse])
async def list_collections(user: dict = Depends(verify_api_key)):
    """List all collections for the authenticated user"""
    user_collections = []
    for collection in database['collections'].values():
        if collection.get("user") == user["name"]:
            user_collections.append(CollectionResponse(
                id=collection["id"],
                name=collection["name"],
                description=collection["description"],
                image_count=collection["image_count"],
                created_date=collection["created_date"],
                model=ModelConfig(**collection["model"])
            ))
    
    return user_collections

@app.post("/collections/{collection_id}/images")
async def add_image_to_collection(
    collection_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(verify_api_key)
):
    """Add an image to a collection using the collection's model"""
    # Verify collection
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Get embedding using collection's model
        model_config = ModelConfig(**collection["model"])
        embedding = model_manager.get_embedding(model_config, image)
        
        # Create image record
        image_id = str(uuid.uuid4())
        image_record = {
            "id": image_id,
            "filename": file.filename,
            "collection_id": collection_id,
            "embedding": embedding,
            "upload_date": datetime.now().isoformat(),
            "user": user["name"],
            "model": collection["model"]
        }
        
        database['images'][image_id] = image_record
        database['collections'][collection_id]['image_count'] += 1
        
        save_database()
        
        return {
            "message": "Image added successfully",
            "image_id": image_id,
            "filename": file.filename,
            "collection_id": collection_id,
            "model_used": f"{model_config.type}:{model_config.name or model_config.endpoint}",
            "embedding_dimensions": len(embedding)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/collections/{collection_id}/images/batch")
async def add_multiple_images_to_collection(
    collection_id: str,
    files: List[UploadFile] = File(...),
    user: dict = Depends(verify_api_key)
):
    """Add multiple images to a collection at once"""
    # Verify collection
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate all files are images
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    results = []
    successful_uploads = 0
    model_config = ModelConfig(**collection["model"])
    
    for file in files:
        try:
            # Process image
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            embedding = model_manager.get_embedding(model_config, image)
            
            # Create image record
            image_id = str(uuid.uuid4())
            image_record = {
                "id": image_id,
                "filename": file.filename,
                "collection_id": collection_id,
                "embedding": embedding,
                "upload_date": datetime.now().isoformat(),
                "user": user["name"],
                "model": collection["model"]
            }
            
            database['images'][image_id] = image_record
            successful_uploads += 1
            
            results.append({
                "filename": file.filename,
                "image_id": image_id,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    # Update collection image count
    database['collections'][collection_id]['image_count'] += successful_uploads
    save_database()
    
    return {
        "message": f"Batch upload completed: {successful_uploads}/{len(files)} images successful",
        "collection_id": collection_id,
        "total_attempted": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": len(files) - successful_uploads,
        "results": results
    }

class BatchUploadFromUrls(BaseModel):
    urls: List[str]

@app.post("/collections/{collection_id}/images/batch-from-urls")
async def add_images_from_urls(
    collection_id: str,
    request: BatchUploadFromUrls,
    user: dict = Depends(verify_api_key)
):
    """Add multiple images to a collection from URLs"""
    # Verify collection
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    results = []
    successful_uploads = 0
    model_config = ModelConfig(**collection["model"])
    
    for url in request.urls:
        try:
            # Download image from URL
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Process image
            image = Image.open(BytesIO(response.content))
            embedding = model_manager.get_embedding(model_config, image)
            
            # Create image record
            image_id = str(uuid.uuid4())
            filename = url.split('/')[-1] or f"image_{image_id[:8]}.jpg"
            
            image_record = {
                "id": image_id,
                "filename": filename,
                "collection_id": collection_id,
                "embedding": embedding,
                "upload_date": datetime.now().isoformat(),
                "user": user["name"],
                "model": collection["model"],
                "source_url": url
            }
            
            database['images'][image_id] = image_record
            successful_uploads += 1
            
            results.append({
                "url": url,
                "filename": filename,
                "image_id": image_id,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })
    
    # Update collection image count
    database['collections'][collection_id]['image_count'] += successful_uploads
    save_database()
    
    return {
        "message": f"Batch URL upload completed: {successful_uploads}/{len(request.urls)} images successful",
        "collection_id": collection_id,
        "total_attempted": len(request.urls),
        "successful_uploads": successful_uploads,
        "failed_uploads": len(request.urls) - successful_uploads,
        "results": results
    }

@app.post("/collections/{collection_id}/search", response_model=SearchResponse)
async def search_similar_images(
    collection_id: str,
    file: UploadFile = File(...),
    top_k: int = 5,
    user: dict = Depends(verify_api_key)
):
    """Search for similar images using the collection's model"""
    import time
    start_time = time.time()
    
    # Verify collection
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process query image
        contents = await file.read()
        query_image = Image.open(BytesIO(contents))
        
        # Get embedding using collection's model
        model_config = ModelConfig(**collection["model"])
        query_embedding = model_manager.get_embedding(model_config, query_image)
        
        # Find similar images
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
        
        # Sort and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results = similarities[:top_k]
        query_time = (time.time() - start_time) * 1000
        
        model_name = f"{model_config.type}:{model_config.name or model_config.endpoint}"
        
        return SearchResponse(
            results=[SearchResult(**result) for result in results],
            total_searched=len(collection_images),
            query_time_ms=round(query_time, 2),
            model_used=model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching images: {str(e)}")

@app.post("/collections/{collection_id}/search-text", response_model=TextSearchResponse)
async def search_similar_images_by_text(
    collection_id: str,
    request: TextSearchRequest,
    user: dict = Depends(verify_api_key)
):
    """Search for images using text query"""
    import time
    start_time = time.time()
    
    # Verify collection
    if collection_id not in database['collections']:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = database['collections'][collection_id]
    if collection.get("user") != user["name"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Get text embedding using collection's model
        model_config = ModelConfig(**collection["model"])
        query_embedding = model_manager.get_text_embedding(model_config, request.query)
        
        # Find similar images
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
        
        # Sort and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results = similarities[:request.top_k]
        query_time = (time.time() - start_time) * 1000
        
        model_name = f"{model_config.type}:{model_config.name or model_config.endpoint}"
        
        return TextSearchResponse(
            results=[SearchResult(**result) for result in results],
            total_searched=len(collection_images),
            query_time_ms=round(query_time, 2),
            model_used=model_name,
            query_text=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching images: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": ["multi_model_support"],
        "builtin_models_loaded": len(model_manager.built_in_models),
        "collections": len(database['collections']) if database else 0,
        "images": len(database['images']) if database else 0
    }

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Visual Search API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔑 Demo API Key: demo_key_12345")
    print("🤖 Built-in Models: clip-vit-large, clip-vit-base")
    print("🔌 Custom Models: Bring your own endpoint")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)