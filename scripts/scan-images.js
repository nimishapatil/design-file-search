#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const https = require('https');

// Configuration
const COHERE_API_KEY = 'npArGaAlzThBw2S9FSH5RcCelsAVIXRy13MH4FVb'; // Replace with your actual API key
const IMAGES_FOLDER = '../test-images';
const DATABASE_FILE = '../database/embeddings.json';

// Supported image formats
const SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'];

// Simple database functions
function loadDatabase() {
    try {
        if (fs.existsSync(DATABASE_FILE)) {
            const data = fs.readFileSync(DATABASE_FILE, 'utf8');
            return JSON.parse(data);
        }
    } catch (error) {
        console.log('Creating new database...');
    }
    return { images: [] };
}

function saveDatabase(db) {
    // Ensure database directory exists
    const dbDir = path.dirname(DATABASE_FILE);
    if (!fs.existsSync(dbDir)) {
        fs.mkdirSync(dbDir, { recursive: true });
    }
    
    fs.writeFileSync(DATABASE_FILE, JSON.stringify(db, null, 2));
}

// Convert image to base64
function imageToBase64(imagePath) {
    const imageBuffer = fs.readFileSync(imagePath);
    return imageBuffer.toString('base64');
}

// Call Cohere API to get image embedding
function getImageEmbedding(base64Image) {
    return new Promise((resolve, reject) => {
        const postData = JSON.stringify({
            model: 'embed-english-v3.0',
            input_type: 'search_document',
            embedding_types: ['float'],
            images: [`data:image/jpeg;base64,${base64Image}`]
        });



        const options = {
            hostname: 'api.cohere.ai',
            port: 443,
            path: '/v1/embed',
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${COHERE_API_KEY}`,
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                try {
                    const response = JSON.parse(data);
                    if (response.embeddings && response.embeddings.float) {
                        resolve(response.embeddings.float[0]);
                    } else {
                        reject(new Error('Invalid response format'));
                    }
                } catch (error) {
                    reject(error);
                }
            });
        });

        req.on('error', (error) => {
            reject(error);
        });

        req.write(postData);
        req.end();
    });
}

// Calculate cosine similarity between two embeddings
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Main scanning function
async function scanImages() {
    console.log('🔍 Starting image scan...');
    
// Check if API key is set
if (!COHERE_API_KEY || COHERE_API_KEY === 'YOUR_API_KEY_HERE') {
    console.error('❌ Please set your Cohere API key in the script!');
    process.exit(1);
}

console.log('✅ API key found, starting scan...');
    
    // Load existing database
    const db = loadDatabase();
    console.log(`📚 Loaded database with ${db.images.length} existing images`);
    
    // Get list of image files
    const imagesPath = path.resolve(__dirname, IMAGES_FOLDER);
    if (!fs.existsSync(imagesPath)) {
        console.error(`❌ Images folder not found: ${imagesPath}`);
        process.exit(1);
    }
    
    const files = fs.readdirSync(imagesPath);
    const imageFiles = files.filter(file => 
        SUPPORTED_FORMATS.includes(path.extname(file))
    );
    
    console.log(`📸 Found ${imageFiles.length} image files`);
    
    let processed = 0;
    let skipped = 0;
    
    for (const filename of imageFiles) {
        const filePath = path.join(imagesPath, filename);
        
        // Check if already processed
        const existingImage = db.images.find(img => img.filename === filename);
        if (existingImage) {
            console.log(`⏭️  Skipping ${filename} (already processed)`);
            skipped++;
            continue;
        }
        
        try {
            console.log(`🔄 Processing ${filename}...`);
            
            // Convert to base64
            const base64Image = imageToBase64(filePath);
            
            // Get embedding from Cohere
            const embedding = await getImageEmbedding(base64Image);
            
            // Add to database
            db.images.push({
                filename: filename,
                path: filePath,
                embedding: embedding,
                processedAt: new Date().toISOString()
            });
            
            // Save database after each image (in case of interruption)
            saveDatabase(db);
            
            processed++;
            console.log(`✅ Processed ${filename} (${processed}/${imageFiles.length})`);
            
            // Small delay to be nice to the API
            await new Promise(resolve => setTimeout(resolve, 1000));
            
        } catch (error) {
            console.error(`❌ Error processing ${filename}:`, error.message);
        }
    }
    
    console.log('\n🎉 Scan complete!');
    console.log(`✅ Processed: ${processed} images`);
    console.log(`⏭️  Skipped: ${skipped} images`);
    console.log(`💾 Database saved to: ${DATABASE_FILE}`);
}

// Export functions for use in other scripts
module.exports = {
    loadDatabase,
    cosineSimilarity,
    getImageEmbedding,
    imageToBase64
};

// Run the scan if this script is executed directly
if (require.main === module) {
    scanImages().catch(console.error);
}