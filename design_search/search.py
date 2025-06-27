"""Main search functions for design_search library"""

from typing import List, Union
from PIL import Image
from .collections import Collection
from .models import ModelConfig

def create_collection(name: str, model: Union[str, ModelConfig] = "clip-vit-base") -> Collection:
    """Create a new image collection
    
    Args:
        name: Collection name
        model: Model to use ("clip-vit-base", "clip-vit-large", or ModelConfig object)
    
    Returns:
        Collection object
        
    Example:
        >>> collection = create_collection("my_images", "clip-vit-base")
        >>> collection.add_image("photo.jpg")
    """
    return Collection(name=name, model_config=model)

def search_images(collection: Collection, query_image: Union[str, Image.Image], top_k: int = 5) -> List[dict]:
    """Search for similar images using an image query
    
    Args:
        collection: Collection to search in
        query_image: Path to image file or PIL Image object
        top_k: Number of results to return
    
    Returns:
        List of search results with similarity scores
        
    Example:
        >>> results = search_images(collection, "query.jpg", top_k=3)
        >>> for result in results:
        ...     print(f"{result['filename']}: {result['similarity']:.2f}")
    """
    return collection.search_by_image(query_image, top_k=top_k)

def search_text(collection: Collection, query: str, top_k: int = 5) -> List[dict]:
    """Search for images using text query
    
    Args:
        collection: Collection to search in  
        query: Text description to search for
        top_k: Number of results to return
    
    Returns:
        List of search results with similarity scores
        
    Example:
        >>> results = search_text(collection, "red shoes", top_k=5)
        >>> for result in results:
        ...     print(f"{result['filename']}: {result['similarity']:.2f}")
    """
    return collection.search_by_text(query, top_k=top_k)