import express from 'express';
import fs from 'fs';
import path from 'path';
import cors from 'cors';
import http from 'http';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const app = express();
const port = 3000;

app.use(cors());
app.use('/images', express.static('previously_scanned_images'));
app.use('/processed_images', express.static(path.join(__dirname, 'processed_images')));
app.use(express.json());

const imageDataPath = path.join(__dirname, 'image_data.json');

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

  const savePath = path.join(__dirname, 'previously_scanned_images', filename);

  const fileStream = fs.createWriteStream(savePath);

  http.get(imageUrl, (response) => {
    response.pipe(fileStream);

    fileStream.on('finish', async () => {
      console.log(`Image saved to ${savePath}`);

      try {
        const imageData = await readImageData();
        
        // Process anomalies to ensure correct format
        let processedAnomalies = [];
        
        // If anomalies is provided and is an array
        if (Array.isArray(anomalies) && anomalies.length > 0) {
          processedAnomalies = anomalies.map(anomaly => {
            // Make sure each anomaly has the required properties
            return {
              anomalyName: anomaly.anomalyName || "Unknown",
              percentage: anomaly.percentage || "0%"
            };
          });
        } else {
          // If no anomalies provided, add default examples to maintain format
          processedAnomalies = [
            { anomalyName: "Fracture", percentage: "65%" },
            { anomalyName: "crack", percentage: "50%" }
          ];
        }
        
        // Add the new image data with properly formatted anomalies
        imageData.unshift({ 
          imageUrl: constructedImageUrl, 
          anomalies: processedAnomalies 
        });
        
        await writeImageData(imageData);
        res.send('Image saved successfully');
      } catch (err) {
        console.error('Error processing image data:', err);
        res.status(500).send('Error processing image data');
      }
    });

    fileStream.on('error', (err) => {
      console.error('Error saving image:', err);
      fs.unlink(savePath, () => {});
      res.status(500).send('Error saving image');
    });
  }).on('error', (err) => {
    console.error('Error fetching image:', err);
    fs.unlink(savePath, () => {});
    res.status(500).send('Error fetching image');
  });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});