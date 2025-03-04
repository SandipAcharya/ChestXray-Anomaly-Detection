import express from 'express';
import fs from 'fs';
import path from 'path';
import cors from 'cors';
import http from 'http';
import { exec } from 'child_process';
import multer from 'multer';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const app = express();
const port = 3000;

// Ensure all required directories exist
const previouslyScannedImagesDir = path.join(__dirname, 'previously_scanned_images');
const processedImagesDir = path.join(__dirname, 'processed_images');
const tempImagesDir = path.join(__dirname, 'temp_images');

[previouslyScannedImagesDir, processedImagesDir, tempImagesDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
});

app.use(cors());

// Important: Configure static file serving with proper MIME types
app.use('/images', express.static(previouslyScannedImagesDir, {
  setHeaders: (res, path) => {
    if (path.endsWith('.jpg') || path.endsWith('.jpeg')) {
      res.setHeader('Content-Type', 'image/jpeg');
    }
  }
}));

app.use('/processed_images', express.static(processedImagesDir, {
  setHeaders: (res, path) => {
    if (path.endsWith('.jpg') || path.endsWith('.jpeg')) {
      res.setHeader('Content-Type', 'image/jpeg');
    }
  }
}));

app.use('/temp_images', express.static(tempImagesDir));

// Increase JSON payload size limit
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Configure multer for file uploads
const upload = multer({ dest: tempImagesDir });

const imageDataPath = path.join(__dirname, 'image_data.json');

// Ensure image_data.json exists
if (!fs.existsSync(imageDataPath)) {
  fs.writeFileSync(imageDataPath, '[]', 'utf8');
  console.log(`Created file: ${imageDataPath}`);
}

const readImageData = async () => {
  try {
    const data = await fs.promises.readFile(imageDataPath, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    console.error('Error reading image_data.json:', err);
    return [];
  }
};

const writeImageData = async (data) => {
  try {
    await fs.promises.writeFile(imageDataPath, JSON.stringify(data, null, 2));
  } catch (err) {
    console.error('Error writing to image_data.json:', err);
  }
};

// List all processed images (for debugging)
app.get('/list-processed-images', (req, res) => {
  try {
    const files = fs.readdirSync(processedImagesDir);
    res.json({ files });
  } catch (err) {
    res.status(500).json({ error: 'Failed to list processed images', details: err.message });
  }
});

app.get('/', async (req, res) => {
  try {
    const imageData = await readImageData();
    res.json(imageData);
  } catch (err) {
    res.status(500).send('Error fetching image data');
  }
});

app.post('/', async (req, res) => {
  const { imageUrl, anomalies } = req.body;

  if (!imageUrl) {
    return res.status(400).send('Image URL is required');
  }

  const filename = path.basename(imageUrl);
  const constructedImageUrl = `http://localhost:${port}/images/${filename}`;

  const savePath = path.join(previouslyScannedImagesDir, filename);

  try {
    // If imageUrl starts with http, download it
    if (imageUrl.startsWith('http')) {
      const fileStream = fs.createWriteStream(savePath);
      
      await new Promise((resolve, reject) => {
        http.get(imageUrl, (response) => {
          response.pipe(fileStream);
          fileStream.on('finish', resolve);
          fileStream.on('error', reject);
        }).on('error', reject);
      });
      
      console.log(`Image saved to ${savePath}`);
    } else {
      // Handle base64 image
      const base64Data = imageUrl.split(',')[1];
      await fs.promises.writeFile(savePath, Buffer.from(base64Data, 'base64'));
      console.log(`Base64 image saved to ${savePath}`);
    }

    // Process and save metadata
    const imageData = await readImageData();
    
    // Process anomalies to ensure correct format
    let processedAnomalies = [];
    
    if (Array.isArray(anomalies) && anomalies.length > 0) {
      processedAnomalies = anomalies.map(anomaly => ({
        anomalyName: anomaly.anomalyName || "Unknown",
        percentage: anomaly.percentage || "0%"
      }));
    } else {
      processedAnomalies = [
        { anomalyName: "Fracture", percentage: "65%" },
        { anomalyName: "crack", percentage: "50%" }
      ];
    }
    
    imageData.unshift({ 
      imageUrl: constructedImageUrl, 
      anomalies: processedAnomalies 
    });
    
    await writeImageData(imageData);
    res.send('Image saved successfully');
  } catch (err) {
    console.error('Error saving image:', err);
    res.status(500).send('Error saving image');
  }
});

// Direct access to a specific processed image (for debugging)
app.get('/processed_images/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(processedImagesDir, filename);
  
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).send('Image not found');
  }
});

// Simplified detection endpoint
app.post('/detect', express.json(), async (req, res) => {
  const { imageUrl, timestamp } = req.body;
  
  if (!imageUrl) {
    return res.status(400).json({ error: 'No image URL provided' });
  }

  // Ensure we have a timestamp
  const processingTimestamp = timestamp || Date.now();
  const outputFilename = `xray_${processingTimestamp}.jpg`;
  
  try {
    // Save the input image to a temporary location
    let imagePath;
    if (imageUrl.startsWith('data:image')) {
      const base64Data = imageUrl.split(',')[1];
      imagePath = path.join(tempImagesDir, outputFilename);
      await fs.promises.writeFile(imagePath, Buffer.from(base64Data, 'base64'));
      console.log('Saved base64 image to:', imagePath);
    } else if (imageUrl.startsWith('http')) {
      imagePath = path.join(tempImagesDir, outputFilename);
      await new Promise((resolve, reject) => {
        const fileStream = fs.createWriteStream(imagePath);
        http.get(imageUrl, (response) => {
          response.pipe(fileStream);
          fileStream.on('finish', resolve);
          fileStream.on('error', reject);
        }).on('error', reject);
      });
      console.log('Downloaded image to:', imagePath);
    } else {
      return res.status(400).json({ error: 'Invalid image URL format' });
    }
    
    // CRITICAL! Save a copy directly to processed_images folder immediately
    // This ensures we have a fallback in case YOLO fails
    const processedImagePath = path.join(processedImagesDir, outputFilename);
    fs.copyFileSync(imagePath, processedImagePath);
    console.log(`Copied image to processed folder: ${processedImagePath}`);
    
    // Respond to the client immediately
    res.json({ 
      message: 'Processing started',
      timestamp: processingTimestamp,
      originalImagePath: imagePath,
      processedImagePath: processedImagePath
    });
    
    // Run YOLOv7 detection in the background (don't wait for it)
    const yoloPath = '/home/sahadev/Downloads/minor/yolov7-main';
    const saveDir = path.resolve(processedImagesDir);
    
    // Ensure the output directory exists
    if (!fs.existsSync(saveDir)) {
      fs.mkdirSync(saveDir, { recursive: true });
    }
    
    // Adjust YOLO command based on your specific setup
    const command = `cd ${yoloPath} && python detect.py --weights xray.pt --source "${imagePath}" --conf-thres 0.25 --img-size 640 --project "${saveDir}" --name "${processingTimestamp}" --exist-ok`;
    
    // Execute YOLO in the background
    console.log(`Executing YOLO command: ${command}`);
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing YOLOv7 detection: ${error}`);
        return;
      }

      console.log(`YOLOv7 detection completed successfully`);
      console.log(`Output: ${stdout}`);
      
      // Check if YOLO created its output with a different path/name
      // If so, try to copy it to our expected location
      try {
        const yoloOutputDir = path.join(saveDir, processingTimestamp.toString());
        if (fs.existsSync(yoloOutputDir)) {
          const files = fs.readdirSync(yoloOutputDir);
          if (files.length > 0) {
            // Get the first file YOLO generated
            const yoloOutputFile = path.join(yoloOutputDir, files[0]);
            // Copy to our expected location
            fs.copyFileSync(yoloOutputFile, processedImagePath);
            console.log(`Copied YOLO output from ${yoloOutputFile} to ${processedImagePath}`);
          }
        }
      } catch (copyErr) {
        console.error(`Error copying YOLO output: ${copyErr}`);
      }
    });
  } catch (err) {
    console.error('Error in detection process:', err);
    res.status(500).json({ error: 'Error processing the image for detection', details: err.message });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
  console.log(`Processed images available at: http://localhost:${port}/processed_images/`);
  console.log(`Previous images available at: http://localhost:${port}/images/`);
});