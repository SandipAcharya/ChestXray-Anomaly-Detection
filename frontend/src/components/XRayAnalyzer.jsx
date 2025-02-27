import React, { useRef, useEffect, useState } from "react";
import DrawingCanvas from "./drawing_canvas";
import path from "path-browserify";

const XRayAnalyzer = () => {
  const [previousScans, setPreviousScans] = useState([]);
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const inputRef = useRef(null);
  const uploadAreaRef = useRef(null);
  const [cardClicked, setCardClicked] = useState(false);
  const [detectClicked, setDetectClicked] = useState(false);
  const [saveClicked, setSaveClicked] = useState(false);
  
  useEffect(() => {
    fetch('http://localhost:3000')  // Node.js server endpoint
      .then(response => response.json())
      .then(data => setPreviousScans(data))
      .catch(error => console.error('Error fetching image paths:', error));
  }, []);
    
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setProcessedImage(null); 
      setCardClicked(false);
      setDetectClicked(false);
      setSaveClicked(true);
      setAnomalies([]);
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
      event.target.value = null;
    }
  };

  const detectAnomalies = () => {
    setTimeout(() => {
      const processedImagePath = "http://localhost:3000/processed_images/processed_image.jpeg"; // Adjust the path as needed
      setProcessedImage(processedImagePath);
      setCardClicked(true);
      setDetectClicked(true);
      setSaveClicked(false);
      setAnomalies([
        { anomalyName: "Fracture", percentage: "85%" },
        { anomalyName: "Infection", percentage: "60%" },
      ]);
    }, 1000);
  };

  const saveDetails = () => {
    if (!saveClicked) {
      if (processedImage) {
        // Fixed: Send the entire anomalies array properly
        fetch('http://localhost:3000', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            imageUrl: processedImage,
            anomalies: anomalies  // Send the full anomalies array
          }),
        })
          .then((response) => response.text())
          .then((data) => {
            console.log(data); // Log success message
            setSaveClicked(true);
            fetch('http://localhost:3000')
            .then((response) => response.json())
            .then((data) => setPreviousScans(data))
            .catch((error) => console.error('Error refetching image paths:', error));
          })
          .catch((error) => console.error('Error saving image:', error));
      } else {
        console.log('No processed image to save.');
      }
    } else {
      console.log("Image already saved.");
    }
  };

  const handleCardClick = (scan) => {
    setImage(scan.imageUrl);
    setProcessedImage(null);
    setCardClicked(true);
    setSaveClicked(true);
    setDetectClicked(false);
    if (scan.anomalies && Array.isArray(scan.anomalies)) {
      setAnomalies(scan.anomalies);
    } else {
      setAnomalies([]); // Clear anomalies if data is missing or invalid
    }
  };

  return (
    <div className="xray-container">
      {/* Left Panel */}
      <div className="left-panel">
        <h1>Previous Scans</h1>
        <div className="scan-list">
          {previousScans
          .filter((scan) => scan.imageUrl) 
          .map((scan, index) => (
            <div
              key={index}
              className="scan-card"
              onClick={() => {
                handleCardClick(scan);
              }}
            >
            {scan.imageUrl && path.basename(scan.imageUrl)}
            {!scan.imageUrl && <p>Image url not found</p>}
            </div>
          ))}
        </div>
      </div>

      {/* Center Panel */}
      <div className="center-panel">
        <input
          type="file"
          accept="image/*"
          ref={inputRef}
          onChange={handleImageUpload}
          hidden
        />

        {!image && (
          <button className="upload-btn" onClick={() => inputRef.current.click()}>
            Upload X-ray
          </button>
        )}

        {(image || processedImage)&& (
          <>
            <div className="upload-area" ref={uploadAreaRef}>
              <img src={processedImage || image} alt="X-ray Image" className="xray-image" />
              <DrawingCanvas image={processedImage || image} uploadAreaRef={uploadAreaRef}/>
            </div>
            <div className="button-group">
              <button className="upload-btn" onClick={() => inputRef.current.click()}>
                Upload
              </button>
              {!cardClicked &&(
              <button className="detect-btn" onClick={detectAnomalies}>
                Detect
              </button>
              )}
              {detectClicked && (
              <button className="save-btn" onClick={saveDetails}>
                {saveClicked ? 'Saved' : 'Save'}
              </button>
              )}
            </div>
          </>
        )}
      </div>

      {/* Right Panel */}
      <div className="right-panel">
        <div className="project_name">X-RAY ANOMALY DETECTION</div>
        <h1>Detection Results</h1>
        {anomalies.length > 0 ? (
          <div className="results">
            {anomalies.map((a, index) => (
              <h2 key={index}>{a.anomalyName}: {a.percentage}</h2>
            ))}
          </div>
        ) : <h2>No anomalies detected</h2>}
      </div>
    </div>
  );
};

export default XRayAnalyzer;