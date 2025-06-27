"""Collection management for design_search library"""

import uuid
import os
from datetime import datetime
from typing import List, Union, Optional
from PIL import Image
from .models import ModelManager, ModelConfig, cosine_similarity

class Collection:
    def __init__(self, name: str, model_config: Union[str, ModelConfig] = "clip-vit-base"):
        """Initialize a new collection
        
        Args:
            name: Collection name
            model_config: Model configuration or model name string
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.created_date = datetime.now().isoformat()
        
        # Handle model config
        if isinstance(model_config, str):
            self.model_config = ModelConfig(type="builtin", name=model_config)
        else:
            self.model_config = model_config
            
        self.model_manager = ModelManager()
        self.images = {}  # Store image data with embeddings
        
    def add_image(self, image_path: str, image_id: Optional[str] = None) -> str:
        """Add a single image to the collection
        
        Args:
            image_path: Path to the image file
            image_id: Optional custom image ID
            
        Returns:
            Image ID
        """
        if image_id is None:
            image_id = str(uuid.uuid4())
            
        # Load and process image
        image = Image.open(image_path)
        embedding = self.model_manager.get_embedding(self.model_config, image)
        
        # Store image data
        self.images[image_id] = {
            "id": image_id,
            "filename": os.path.basename(image_path),
            "path": image_path,
            "embedding": embedding,
            "upload_date": datetime.now().isoformat()
        }
        
        return image_id
    
    def add_images(self, image_paths: List[str]) -> List[str]:
        """Add multiple images to the collection
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of image IDs
        """
        image_ids = []
        for image_path in image_paths:
            try:
                image_id = self.add_image(image_path)
                image_ids.append(image_id)
                print(f"✅ Added {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Failed to add {os.path.basename(image_path)}: {e}")
                
        return image_ids
    
    def search_by_image(self, query_image: Union[str, Image.Image], top_k: int = 5) -> List[dict]:
        """Search for similar images using an image query
        
        Args:
            query_image: Path to image file or PIL Image object
            top_k: Number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        # Load query image if path provided
        if isinstance(query_image, str):
            query_image = Image.open(query_image)
            
        # Get query embedding
        query_embedding = self.model_manager.get_embedding(self.model_config, query_image)
        
        # Calculate similarities
        results = []
        for image_data in self.images.values():
            similarity = cosine_similarity(query_embedding, image_data['embedding'])
            results.append({
                'image_id': image_data['id'],
                'filename': image_data['filename'],
                'path': image_data['path'],
                'similarity': similarity
            })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for images using text query
        
        Args:
            query: Text description to search for
            top_k: Number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        # Get text embedding
        query_embedding = self.model_manager.get_text_embedding(self.model_config, query)
        
        # Calculate similarities
        results = []
        for image_data in self.images.values():
            similarity = cosine_similarity(query_embedding, image_data['embedding'])
            results.append({
                'image_id': image_data['id'],
                'filename': image_data['filename'], 
                'path': image_data['path'],
                'similarity': similarity,
                'query': query
            })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def list_images(self) -> List[dict]:
        """List all images in the collection
        
        Returns:
            List of image metadata (without embeddings)
        """
        return [
            {
                'image_id': img['id'],
                'filename': img['filename'],
                'path': img['path'],
                'upload_date': img['upload_date']
            }
            for img in self.images.values()
        ]
    
    def get_stats(self) -> dict:
        """Get collection statistics
        
        Returns:
            Dictionary with collection stats
        """
        return {
            'id': self.id,
            'name': self.name,
            'created_date': self.created_date,
            'image_count': len(self.images),
            'model': {
                'type': self.model_config.type,
                'name': self.model_config.name
            }
        }
    
    def __repr__(self):
        return f"Collection(name='{self.name}', images={len(self.images)}, model='{self.model_config.name}')"