#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const https = require('https');

// Replace with your actual API key
const COHERE_API_KEY = 'npArGaAlzThBw2S9FSH5RcCelsAVIXRy13MH4FVb';

function imageToBase64(imagePath) {
    const imageBuffer = fs.readFileSync(imagePath);
    return imageBuffer.toString('base64');
}

function testAPI() {
    // Find the first image in your test-images folder
    const imagesPath = path.resolve(__dirname, '../test-images');
    const files = fs.readdirSync(imagesPath);
    const imageFile = files.find(file => 
        ['.png', '.jpg', '.jpeg'].includes(path.extname(file).toLowerCase())
    );
    
    if (!imageFile) {
        console.error('No image files found in test-images folder');
        return;
    }
    
    console.log('Testing with image:', imageFile);
    
    const imagePath = path.join(imagesPath, imageFile);
    const base64Image = imageToBase64(imagePath);
    
    console.log('Image size:', base64Image.length, 'characters');
    console.log('First 50 chars of base64:', base64Image.substring(0, 50));
    
    // Test different API configurations
    const tests = [
        {
            name: 'v2/embed with search_document',
            path: '/v2/embed',
            body: {
                model: 'embed-english-v3.0',
                input_type: 'search_document',
                embedding_types: ['float'],
                images: [`data:image/jpeg;base64,${base64Image}`]
            }
        },
        {
            name: 'v1/embed with image input_type',
            path: '/v1/embed',
            body: {
                model: 'embed-english-v3.0',
                input_type: 'image',
                embedding_types: ['float'],
                images: [base64Image]
            }
        },
        {
            name: 'v2/embed with embed-v4.0',
            path: '/v2/embed',
            body: {
                model: 'embed-v4.0',
                input_type: 'search_document',
                embedding_types: ['float'],
                images: [`data:image/jpeg;base64,${base64Image}`]
            }
        }
    ];
    
    async function runTest(test) {
        return new Promise((resolve) => {
            console.log(`\n🧪 Testing: ${test.name}`);
            
            const postData = JSON.stringify(test.body);
            
            const options = {
                hostname: 'api.cohere.ai',
                port: 443,
                path: test.path,
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
                    console.log('Status:', res.statusCode);
                    console.log('Headers:', JSON.stringify(res.headers, null, 2));
                    
                    try {
                        const response = JSON.parse(data);
                        console.log('Response:', JSON.stringify(response, null, 2));
                        
                        if (response.embeddings && response.embeddings.float) {
                            console.log('✅ SUCCESS! Got embeddings, length:', response.embeddings.float[0].length);
                        } else {
                            console.log('❌ No embeddings in response');
                        }
                    } catch (error) {
                        console.log('❌ Failed to parse JSON response:', data);
                    }
                    resolve();
                });
            });
            
            req.on('error', (error) => {
                console.log('❌ Network error:', error.message);
                resolve();
            });
            
            req.write(postData);
            req.end();
        });
    }
    
    async function runAllTests() {
        for (const test of tests) {
            await runTest(test);
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds between tests
        }
    }
    
    runAllTests();
}

testAPI();