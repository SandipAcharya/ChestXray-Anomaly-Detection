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
  const [isLoading, setIsLoading] = useState(false);
  const [processingTimestamp, setProcessingTimestamp] = useState(null);
  const [renamingScan, setRenamingScan] = useState(null);
  const [newScanName, setNewScanName] = useState("");
  const [contextMenu, setContextMenu] = useState(null);
  // Load previous scans on component mount
  useEffect(() => {
    fetch('http://localhost:3000')
      .then(response => response.json())
      .then(data => setPreviousScans(data))
      .catch(error => console.error('Error fetching image paths:', error));
  }, []);

  const handleRightClick = (event, scan) => {
    event.preventDefault();
    setContextMenu({
      scan,
      x: event.clientX + "px",
      y: event.clientY + "px",
    });
  };
  
  const handleClickOutside = (event) => {
    if (!event.target.closest(".context-menu")) {
      setContextMenu(null);
    }
  };
  
  // Close menu when clicking outside
  useEffect(() => {
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);
  

  // Handle the detection delay and image loading
  useEffect(() => {
    let timer;

    if (detectClicked && processingTimestamp) {
        console.log(`Waiting for processed image with timestamp: ${processingTimestamp}`);

        // Set a fixed delay to wait for the image processing
        timer = setTimeout(async () => {
            // Construct URLs for processed image and anomalies JSON
            const expectedImageUrl = `http://localhost:3000/processed_images/xray_${processingTimestamp}.jpg`;
            const jsonFileUrl = `http://localhost:3000/processed_images/anomalies.json`;

            console.log(`Attempting to load processed image from: ${expectedImageUrl}`);
            console.log(`Fetching anomalies from: ${jsonFileUrl}`);

            // Set the processed image URL
            setProcessedImage(expectedImageUrl);
            setIsLoading(false);

            try {
                // Fetch anomalies JSON file
                const response = await fetch(jsonFileUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch anomalies JSON: ${response.statusText}`);
                }
                const jsonData = await response.json();
                
                // Set anomalies from fetched JSON data
                setAnomalies(jsonData);
            } catch (error) {
                console.error('Error fetching anomalies JSON:', error);

                // Set default anomalies if fetching fails
                setAnomalies([
                ]);
            }
        }, 8000); // Wait 7 seconds to give YOLO enough time
    }

    return () => {
        if (timer) clearTimeout(timer);
    };
}, [detectClicked, processingTimestamp]);


    
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setProcessedImage(null); 
      setCardClicked(false);
      setDetectClicked(false);
      setSaveClicked(false);
      setProcessingTimestamp(null);
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
      event.target.value = null;
    }
  };

  const detectAnomalies = () => {
    if (!image) {
      console.log('No image to process');
      return;
    }
  
    console.log('Starting anomaly detection');
    setIsLoading(true);
    setDetectClicked(true);
    
    // Generate a timestamp for this processing job
    const timestamp = Date.now();
    setProcessingTimestamp(timestamp);
    
    // Create a lower quality version to reduce payload size
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Set to a reasonable size that maintains enough detail for detection
      const maxWidth = 800;
      const maxHeight = 800;
      let width = img.width;
      let height = img.height;
      
      if (width > height) {
        if (width > maxWidth) {
          height *= maxWidth / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width *= maxHeight / height;
          height = maxHeight;
        }
      }
      
      canvas.width = width;
      canvas.height = height;
      
      ctx.drawImage(img, 0, 0, width, height);
      
      // Create lower quality JPEG (0.7 quality)
      const compressedImage = canvas.toDataURL('image/jpeg', 0.7);
      
      // Send the image with the timestamp in the request body
      fetch('http://localhost:3000/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          imageUrl: compressedImage,
          timestamp: timestamp
        }),
      })
      .then(response => {
        console.log('Received response status:', response.status);
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Response from detect endpoint:', data);
        // We don't set anything here - we'll wait for the useEffect to load the image after the delay
      })
      .catch(error => {
        console.error('Error detecting anomalies:', error);
        setIsLoading(false);
      });
    };
    
    img.src = image;
  };

  const saveDetails = () => {
    if (!saveClicked) {
      if (processedImage) {
        fetch('http://localhost:3000', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            imageUrl: processedImage,
            anomalies: anomalies
          }),
        })
          .then((response) => response.text())
          .then((data) => {
            console.log(data);
            setSaveClicked(true);
            // Refresh the previous scans list
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
      setAnomalies([]);
    }
  };

  const handleRename = (scan) => {
    setRenamingScan(scan);
    setNewScanName(path.basename(scan.imageUrl, path.extname(scan.imageUrl)));
  };


  const renameScan = () => {
    if (!renamingScan || !newScanName.trim()) return;
    
    fetch("http://localhost:3000/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ oldName: renamingScan.imageUrl, newName: newScanName }),
    })
      .then((response) => response.text())
      .then(() => {
        setPreviousScans((prevScans) =>
          prevScans.map((scan) =>
            scan.imageUrl === renamingScan.imageUrl ? { ...scan, imageUrl: newScanName } : scan
          )
        );
        setRenamingScan(null);
      })
      .catch((error) => console.error("Error renaming scan:", error));
  };

  const handleDeleteScan = (scan) => {
    if (!scan || !scan.imageUrl) return;
  
    const confirmed = window.confirm("Are you sure you want to delete this scan?");
    if (!confirmed) return;
  
    fetch("http://localhost:3000/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUrl: scan.imageUrl }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to delete scan");
        }
        return response.text();
      })
      .then(() => {
        setPreviousScans((prevScans) =>
          prevScans.filter((item) => item.imageUrl !== scan.imageUrl)
        );
      })
      .catch((error) => console.error("Error deleting scan:", error));
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
          onClick={() => handleCardClick(scan)}
          onContextMenu={(e) => handleRightClick(e, scan)}
        >
          {path.basename(scan.imageUrl)}

          {/* Context Menu */}
          {contextMenu && contextMenu.scan === scan && (
            <div 
              className="context-menu" 
              style={{ top: contextMenu.y, left: contextMenu.x }}
            >
              <button onClick={() => handleRename(scan)}>Rename</button>
              <button onClick={() => handleDeleteScan(scan)}>Delete</button>
            </div>
          )}
        </div>
      ))}
  </div>

  {/* Rename Dialog */}
  {renamingScan && (
    <div className="rename-dialog">
      <input
        type="text"
        value={newScanName}
        onChange={(e) => setNewScanName(e.target.value)}
      />
      <button onClick={renameScan}>Save</button>
      <button onClick={() => setRenamingScan(null)}>Cancel</button>
    </div>
  )}
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

        {(image || processedImage) && (
          <>
            <div className="upload-area" ref={uploadAreaRef}>
              {isLoading ? (
                <div className="loading">
                  Processing image... 
                  <div className="loading-spinner"></div>
                </div>
              ) : (
                <>
                  <img 
                    src={processedImage || image} 
                    alt="X-ray Image" 
                    className="xray-image" 
                    onError={(e) => {
                      console.error("Image failed to load:", e.target.src);
                      e.target.src = image; // Fallback to original image
                    }}
                  />
                  <DrawingCanvas image={processedImage || image} uploadAreaRef={uploadAreaRef}/>
                </>
              )}
            </div>
            <div className="button-group">
              <button className="upload-btn" onClick={() => inputRef.current.click()}>
                Upload
              </button>
              {!cardClicked && !isLoading && (
                <button className="detect-btn" onClick={detectAnomalies}>
                  Detect
                </button>
              )}
              {detectClicked && processedImage && !isLoading && (
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
        {isLoading ? (
          <div>Processing...</div>
        ) : anomalies.length > 0 ? (
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