# AI-Powered Search API

> Add intelligent image and text search to your app in minutes

Transform any application with AI-powered search that works with both images and text queries. Perfect for e-commerce, design tools, asset management, and content discovery.

---

## 🚀 Quick Start

### 1. Start the API
```bash
git clone [your-repo-url]
cd design-search
source clip-env/bin/activate
cd scripts
uvicorn main-byom:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Test It
Visit: **http://localhost:8000/docs** for interactive documentation

---

## ✨ Key Features

- **🔍 Dual Search Modes**: Search with images OR text
- **📷 Image-to-Image**: Find visually similar images
- **💬 Text-to-Image**: Search images using natural language
- **🚀 Batch Upload**: Add multiple images at once
- **🔧 Multi-Model Support**: Built-in CLIP models + bring your own
- **⚡ Fast & Reliable**: Load tested for production use

---

## 📖 Basic Usage

### Create a Collection
```bash
curl -X POST "http://localhost:8000/collections" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Images",
    "description": "AI search collection",
    "model": {"type": "builtin", "name": "clip-vit-base"}
  }'
```

### Upload Images (Single)
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/images" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "file=@path/to/your/image.jpg"
```

### Upload Images (Batch)
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/images/batch" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Search with Images
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/search" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "file=@query_image.jpg"
```

### Search with Text
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/search-text" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "red shoes", "top_k": 5}'
```

---

## 💻 Code Examples

### Node.js
```javascript
const FormData = require('form-data');

// Text search
async function searchByText(collectionId, query) {
  const response = await fetch(`http://localhost:8000/collections/${collectionId}/search-text`, {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer demo_key_12345',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query, top_k: 5 })
  });
  return response.json();
}

// Image search
async function searchByImage(collectionId, imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch(`http://localhost:8000/collections/${collectionId}/search`, {
    method: 'POST',
    headers: { 'Authorization': 'Bearer demo_key_12345' },
    body: formData
  });
  return response.json();
}
```

### Python
```python
import requests

def search_by_text(collection_id, query):
    response = requests.post(
        f'http://localhost:8000/collections/{collection_id}/search-text',
        headers={'Authorization': 'Bearer demo_key_12345'},
        json={'query': query, 'top_k': 5}
    )
    return response.json()

def search_by_image(collection_id, image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f'http://localhost:8000/collections/{collection_id}/search',
            headers={'Authorization': 'Bearer demo_key_12345'},
            files=files
        )
    return response.json()
```

### React
```jsx
function AISearch({ collectionId }) {
  const [results, setResults] = useState([]);
  const [searchMode, setSearchMode] = useState('text');

  const searchByText = async (query) => {
    const response = await fetch(`/collections/${collectionId}/search-text`, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer demo_key_12345',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query, top_k: 5 })
    });
    const data = await response.json();
    setResults(data.results);
  };

  const searchByImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`/collections/${collectionId}/search`, {
      method: 'POST',
      headers: { 'Authorization': 'Bearer demo_key_12345' },
      body: formData
    });
    const data = await response.json();
    setResults(data.results);
  };

  return (
    <div>
      <div>
        <button onClick={() => setSearchMode('text')}>Text Search</button>
        <button onClick={() => setSearchMode('image')}>Image Search</button>
      </div>
      
      {searchMode === 'text' ? (
        <input 
          type="text" 
          placeholder="Search for 'red shoes', 'logo', etc."
          onKeyPress={(e) => e.key === 'Enter' && searchByText(e.target.value)}
        />
      ) : (
        <input 
          type="file" 
          accept="image/*"
          onChange={(e) => searchByImage(e.target.files[0])} 
        />
      )}
      
      {results.map(result => (
        <div key={result.image_id}>
          {result.filename} - {(result.similarity * 100).toFixed(1)}% match
        </div>
      ))}
    </div>
  );
}
```

---

## 🔧 Model Options

### Built-in Models (Free)
- **clip-vit-base**: Fast, good accuracy (512 dimensions)
- **clip-vit-large**: Slower, higher accuracy (768 dimensions)

### Bring Your Own Model (BYOM)
```json
{
  "model": {
    "type": "endpoint",
    "endpoint": "https://your-model-api.com/embed",
    "api_key": "your-api-key"
  }
}
```

---

## 📊 Response Format

### Search Results
```json
{
  "results": [
    {
      "image_id": "uuid",
      "filename": "product.jpg", 
      "similarity": 0.87,
      "collection_id": "collection-uuid"
    }
  ],
  "total_searched": 25,
  "query_time_ms": 145.2,
  "model_used": "builtin:clip-vit-base",
  "query_text": "red shoes"
}
```

---

## 🛠️ API Endpoints

### Collections
- `POST /collections` - Create collection with model choice
- `GET /collections` - List your collections
- `DELETE /collections/{id}` - Delete collection

### Images  
- `POST /collections/{id}/images` - Upload single image
- `POST /collections/{id}/images/batch` - Upload multiple images
- `GET /collections/{id}/images` - List images in collection

### Search
- `POST /collections/{id}/search` - Search by image similarity
- `POST /collections/{id}/search-text` - Search by text query

### System
- `GET /health` - Health check
- `GET /models` - Available models

**Full docs**: http://localhost:8000/docs

---

## 💡 Use Cases

### E-commerce
- **"Find similar products"** with image upload
- **"red dress"** or **"vintage shoes"** with text search
- **Product discovery** and recommendation engines

### Design Tools
- **"Find similar logos"** by uploading a design
- **"minimalist icons"** or **"dark theme UI"** with text
- **Asset organization** and design system management

### Content Management
- **"Find team photos"** or **"product shots"** with text
- **Similar image detection** for deduplication
- **Visual content discovery** and organization

### Development
- **"Add visual search"** to any application
- **No ML expertise required** - just HTTP calls
- **Production-ready** with load testing completed

---

## 🚀 Performance

- **Response Time**: ~150ms average search time
- **Concurrent Users**: Tested with 5+ simultaneous users
- **Scalability**: Ready for production deployment
- **Reliability**: 100% uptime in load testing

---

## 🔑 Authentication

Get your API key by signing up at [your-signup-url]

Demo key for testing: `demo_key_12345`

---

## 🐛 Issues & Feedback

Found a bug or have feedback? We'd love to hear from you!

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Email**: hello@swirlypeak.com

---

**Ready to add AI-powered search to your app?** Clone the repo and start building!# Visual Search API

> Find visually similar images using AI embeddings

Add image similarity search to your app in minutes. Perfect for design tools, asset management, and creative applications.

---

## 🚀 Quick Start

### 1. Start the API
```bash
```bash
git clone [repo-url]
cd design-search
source clip-env/bin/activate
cd scripts
uvicorn main-byom:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Test It
Visit: **http://localhost:8080/docs** for interactive documentation

---

## 📖 Basic Usage

### Create a Collection
```bash
curl -X POST "http://localhost:8080/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Images",
    "description": "Test collection"
  }'
```

### Upload Images
```bash
curl -X POST "http://localhost:8080/collections/{collection_id}/images" \
  -F "file=@path/to/your/image.jpg"
```

### Search for Similar Images
```bash
curl -X POST "http://localhost:8080/collections/{collection_id}/search" \
  -F "file=@path/to/query/image.jpg"
```

---

## 💻 Code Examples

### Node.js
```javascript
const FormData = require('form-data');

async function searchSimilar(collectionId, imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch(`http://localhost:8080/collections/${collectionId}/search`, {
    method: 'POST',
    body: formData
  });
  
  return response.json();
}
```

### Python
```python
import requests

def search_similar(collection_id, image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f'http://localhost:8080/collections/{collection_id}/search',
            files=files
        )
    return response.json()
```

### React
```jsx
function ImageSearch({ collectionId }) {
  const [results, setResults] = useState([]);
  
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`/collections/${collectionId}/search`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setResults(data.results);
  };
  
  return (
    <div>
      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />
      {results.map(result => (
        <div key={result.image_id}>
          {result.filename} - {(result.similarity * 100).toFixed(1)}% match
        </div>
      ))}
    </div>
  );
}
```

---

## 📊 Response Format

### Search Results
```json
{
  "results": [
    {
      "image_id": "uuid",
      "filename": "design.png", 
      "similarity": 0.87
    }
  ]
}
```

---

## 🛠️ API Endpoints

- `POST /collections` - Create image collection
- `POST /collections/{id}/images` - Upload images (max 5MB, JPEG/PNG/WebP)
- `POST /collections/{id}/search` - Find similar images
- `GET /health` - Health check
- `GET /models` – Choose models

**Full docs**: http://localhost:8080/docs

---

## 💡 Use Cases

- **Design Tools**: Find similar design components
- **E-commerce**: "Find similar products" feature  
- **Asset Management**: Organize media libraries
- **Content Apps**: Visual content discovery

---

## 🐛 Issues & Feedback

Found a bug or have feedback? We'd love to hear from you!

- **Email**: hello@swirlypeak.com

---

**Ready to test?** Clone the repo and start experimenting with visual search!
