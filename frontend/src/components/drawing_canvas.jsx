import React, { useRef, useEffect } from "react";
const DrawingCanvas = ({ image, uploadAreaRef }) => {
    const canvasRef = useRef(null);
    const isDrawing = useRef(false);
    const lastPoint = useRef({ x: 0, y: 0 }); 
    
    useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const uploadArea = uploadAreaRef.current;
    if (!uploadArea) return;

    canvas.width = uploadArea.clientWidth;
    canvas.height = uploadArea.clientHeight;

    if (image) {
      const img = new Image();
      img.src = image;
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
    }

    const startDrawing = (e) => {
      if (e.button === 0) { 
        isDrawing.current = true;
        lastPoint.current = { x: e.offsetX, y: e.offsetY };
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
      }
    };

    const draw = (e) => {
      if (!isDrawing.current) return;
      const { x, y } = lastPoint.current;
      const newX = e.offsetX;
      const newY = e.offsetY;
      ctx.lineTo(newX, newY);
      ctx.strokeStyle = "red"; // Drawing color
      ctx.lineWidth = 10;
      ctx.stroke();
      lastPoint.current = { x: newX, y: newY };
    };

    const stopDrawing = (e) => {
      if (e.button === 0) { // Ensure only the left button triggers the clear
        isDrawing.current = false;
    
        // Clear the canvas when the left mouse button is released
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          ctx.clearRect(0, 0, canvas.width, canvas.height); // Clears the entire drawing
        }
      }
    };


    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    return () => {
      canvas.removeEventListener("mousedown", startDrawing);
      canvas.removeEventListener("mousemove", draw);
      canvas.removeEventListener("mouseup", stopDrawing);
      canvas.removeEventListener("mouseleave", stopDrawing);
      canvas.removeEventListener("contextmenu", (e) => e.preventDefault());
    };
  }, [image]);
  return <canvas ref={canvasRef} className="draw-canvas"></canvas>;
};
export default DrawingCanvas;