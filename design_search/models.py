"""Model management for design_search library"""

import requests
import base64
from typing import List, Optional, Dict, Any
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pydantic import BaseModel, validator
from fastapi import HTTPException

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
    
    @validator('name')
    def validate_builtin_name(cls, v, values):
        if values.get('type') == 'builtin':
            if v is None:
                raise ValueError('Built-in models require a name')
            if v not in ['clip-vit-base', 'clip-vit-large']:
                raise ValueError('Built-in model name must be: clip-vit-base or clip-vit-large')
        return v
    
    @validator('endpoint')
    def validate_endpoint_url(cls, v, values):
        if values.get('type') == 'endpoint' and not v:
            raise ValueError('Endpoint models require a URL')
        return v
    
    # Convenience class methods
    @classmethod
    def builtin(cls, model_name: str = "clip-vit-base"):
        """Create a built-in model config
        
        Args:
            model_name: Either 'clip-vit-base' or 'clip-vit-large'
        """
        return cls(type="builtin", name=model_name)
    
    @classmethod  
    def endpoint(cls, url: str, api_key: str = None):
        """Create an endpoint model config
        
        Args:
            url: API endpoint URL
            api_key: Optional API key for authentication
        """
        return cls(type="endpoint", endpoint=url, api_key=api_key)

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
    
    def get_endpoint_text_embedding(self, endpoint: str, api_key: str, text: str) -> List[float]:
        """Get text embedding from external endpoint"""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {"text": text, "type": "text"}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "embedding" not in result:
                raise ValueError("External model must return {'embedding': [...]}")
            
            return result["embedding"]
        
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"External model error: {str(e)}")
    
    def get_openai_embedding(self, api_key: str, image: Image.Image) -> List[float]:
        """Get embedding from OpenAI (placeholder for future)"""
        raise HTTPException(status_code=501, detail="OpenAI integration coming soon")
    
    def get_openai_text_embedding(self, api_key: str, text: str) -> List[float]:
        """Get text embedding from OpenAI (placeholder for future)"""
        raise HTTPException(status_code=501, detail="OpenAI text integration coming soon")
    
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

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))