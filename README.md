# Visual Search API

> Find visually similar images using AI embeddings

Add image similarity search to your app in minutes. Perfect for design tools, asset management, and creative applications.

---

## 🚀 Quick Start

### 1. Start the API
```bash
git clone [your-repo-url]
cd design-search
source clip-env/bin/activate
uvicorn scripts.main-byom:app --reload --port 8080
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

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Email**: hello@swirlypeak.com

---

**Ready to test?** Clone the repo and start experimenting with visual search!
