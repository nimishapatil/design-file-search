# API Server Example

This example shows how to create a REST API service using the design_search library.

## Quick Start

```bash
# From the design-search root directory
cd examples/api_server

# Make sure you're in the virtual environment
source ../../clip-env/bin/activate

# Start the API server
./start-api.sh
```

## API Documentation

Visit: http://localhost:8000/docs

## Key Features

- **REST API** built with FastAPI
- **Multi-model support** using design_search library
- **Image and text search** endpoints
- **Batch upload** capabilities
- **Authentication** with API keys

## Endpoints

- `POST /collections` - Create image collection
- `POST /collections/{id}/images` - Upload single image
- `POST /collections/{id}/images/batch` - Upload multiple images  
- `POST /collections/{id}/search` - Search by image
- `POST /collections/{id}/search-text` - Search by text
- `GET /health` - Health check

## Usage Examples

### Create Collection
```bash
curl -X POST "http://localhost:8000/collections" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Images",
    "model": {"type": "builtin", "name": "clip-vit-base"}
  }'
```

### Text Search
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/search-text" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "mountain landscape", "top_k": 5}'
```

## Compared to Library Usage

**API Server (this example):**
- HTTP endpoints for web applications
- Multi-user with authentication
- Centralized deployment
- Language agnostic (any language can call HTTP APIs)

**Python Library:**
- Direct function calls
- Single-user, embedded in application
- Distributed deployment
- Python-specific

Choose the approach that fits your use case!