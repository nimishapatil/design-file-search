# AI-Powered Visual Search API

> Add intelligent image and text search to your applications in minutes

Transform any application with AI-powered search that works with both images and text queries. Perfect for design tools, e-commerce, asset management, and creative applications.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
git clone [nimishapatil/design-file-search]
cd design-search

# Create and activate virtual environment
python -m venv clip-env
source clip-env/bin/activate  # On Windows: clip-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API
```bash
cd scripts
python main-byom.py
```

### 3. Test It
Visit: **http://localhost:8000/docs** for interactive API documentation

---

## ✨ Key Features

- **🔍 Dual Search Modes**: Search with images OR text queries
- **📷 Image-to-Image**: Find visually similar images
- **💬 Text-to-Image**: Search images using natural language descriptions
- **🚀 Batch Operations**: Upload multiple images at once or from URLs
- **🔧 Multi-Model Support**: Built-in CLIP models + bring your own model endpoints
- **⚡ Production Ready**: Load tested with monitoring and health checks
- **🐍 Python Library**: Use directly in Python applications
- **🌐 REST API**: HTTP endpoints for any programming language

---

## 📖 API Usage

### Authentication
All API requests require an API key in the Authorization header:
```bash
-H "Authorization: Bearer demo_key_12345"
```

### Create a Collection
```bash
curl -X POST "http://localhost:8000/collections" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Design Files",
    "description": "Collection for design assets",
    "model": {"type": "builtin", "name": "clip-vit-base"}
  }'
```

### Upload Images

**Single Image:**
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/images" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "file=@path/to/your/image.jpg"
```

**Batch Upload (Multiple Files):**
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/images/batch" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Batch Upload from URLs:**
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/images/batch-from-urls" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg"
    ]
  }'
```

### Search

**Search by Image:**
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/search" \
  -H "Authorization: Bearer demo_key_12345" \
  -F "file=@query_image.jpg" \
  -F "top_k=5"
```

**Search by Text:**
```bash
curl -X POST "http://localhost:8000/collections/{collection_id}/search-text" \
  -H "Authorization: Bearer demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "red logo design", "top_k": 5}'
```

---

## 🐍 Python Library Usage

Use the library directly in your Python applications:

```python
from design_search import create_collection, search_images, search_text

# Create a collection
collection = create_collection("my_designs", model="clip-vit-base")

# Add images
collection.add_image("design1.jpg")
collection.add_images(["design2.jpg", "design3.jpg", "design4.jpg"])

# Search by image
results = search_images(collection, "query_design.jpg", top_k=3)
for result in results:
    print(f"{result['filename']}: {result['similarity']:.2f}")

# Search by text
results = search_text(collection, "minimalist logo", top_k=3)
for result in results:
    print(f"{result['filename']}: {result['similarity']:.2f}")
```

---

## 💻 Code Examples

### JavaScript/Node.js
```javascript
const FormData = require('form-data');

// Create collection
async function createCollection(name) {
  const response = await fetch('http://localhost:8000/collections', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer demo_key_12345',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      name: name,
      model: { type: "builtin", name: "clip-vit-base" }
    })
  });
  return response.json();
}

// Search by text
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
```

### React Component
```jsx
function VisualSearch({ collectionId }) {
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
          placeholder="Search for 'logo', 'mountain', etc."
          onKeyPress={(e) => e.key === 'Enter' && searchByText(e.target.value)}
        />
      ) : (
        <input 
          type="file" 
          accept="image/*"
          onChange={(e) => searchByImage(e.target.files[0])} 
        />
      )}
      
      <div>
        {results.map(result => (
          <div key={result.image_id}>
            {result.filename} - {(result.similarity * 100).toFixed(1)}% match
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 🔧 Model Options

### Built-in Models (Free)
- **clip-vit-base**: Fast processing, good accuracy (512 dimensions)
- **clip-vit-large**: Slower processing, higher accuracy (768 dimensions)

### Custom Models (Bring Your Own)
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
      "image_id": "uuid-string",
      "filename": "design.jpg", 
      "similarity": 0.87,
      "collection_id": "collection-uuid"
    }
  ],
  "total_searched": 25,
  "query_time_ms": 145.2,
  "model_used": "builtin:clip-vit-base",
  "query_text": "red logo design"
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
- `POST /collections/{id}/images/batch-from-urls` - Upload from URLs

### Search
- `POST /collections/{id}/search` - Search by image similarity
- `POST /collections/{id}/search-text` - Search by text query

### System
- `GET /health` - Health check
- `GET /models` - Available models

**Complete documentation**: http://localhost:8000/docs

---

## 💡 Use Cases

### Design Tools & Creative Apps
- **Find similar design components** by uploading a reference
- **"minimalist logo"** or **"dark theme UI"** with text search
- **Design system management** and asset organization
- **Version control** for design iterations

### E-commerce & Product Discovery
- **"Find similar products"** with image upload
- **"red dress"** or **"vintage shoes"** with text search
- **Product recommendation** engines
- **Visual merchandising** tools

### Content Management & Media
- **"Find team photos"** or **"product shots"** with text
- **Duplicate detection** and content deduplication
- **Asset library organization** with AI categorization
- **Stock photo** and media discovery

### Development & Integration
- **Add visual search** to any application via REST API
- **No ML expertise required** - just HTTP calls
- **Multi-language support** - works with any programming language
- **Production-ready** with authentication and monitoring

---

## 🚀 Performance & Production

- **Response Time**: ~150ms average search time
- **Concurrent Users**: Tested with 5+ simultaneous users
- **Load Testing**: Included monitoring and stress testing tools
- **Scalability**: Ready for production deployment
- **Reliability**: Health checks and error handling

### Load Testing
```bash
cd load-testing
python simple_test.py
```

### System Monitoring
```bash
cd load-testing  
python monitor.py
```

---

## 🔄 Development & Testing

### Run Examples
```bash
# Python library example
cd examples/simple_search
python basic_example.py

# API server example  
cd examples/api_server
./start-api.sh
```

### File Structure
```
design-search/
├── design_search/           # Python library
│   ├── __init__.py
│   ├── collections.py       # Collection management
│   ├── models.py           # Model handling
│   └── search.py           # Search functions
├── scripts/                # API server
│   ├── main-byom.py        # Multi-model API
│   └── start-api.sh
├── examples/               # Usage examples
├── load-testing/           # Performance testing
└── requirements.txt        # Dependencies
```

---

## 🔑 Authentication

Current demo key: `demo_key_12345`

For production deployment, implement proper API key management and user authentication.

---

## 🐛 Issues & Support

Found a bug or need help?

- **Create an issue** on GitHub
- **Email**: hello@swirlypeak.com

---

## 🎯 Next Steps

1. **Try the demo**: Start the API and visit `/docs`
2. **Upload test images**: Use the batch upload endpoints
3. **Test searches**: Try both image and text queries
4. **Integration**: Add to your application using the code examples
5. **Production**: Deploy with proper authentication and monitoring

**Ready to add AI-powered search to your app?** Start with the quick setup guide above!