"""
Design Search - AI-Powered Search Library
Add intelligent image and text search to your applications.
"""

from .collections import Collection
from .models import ModelManager, ModelConfig, cosine_similarity
from .search import search_images, search_text, create_collection

__version__ = "0.1.0"
__all__ = [
    "Collection",
    "ModelManager", 
    "ModelConfig",
    "cosine_similarity",  # Add this line
    "search_images",
    "search_text",
    "create_collection"
]