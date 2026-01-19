"""
Lightweight inference server for Raspberry Pi / Edge devices.
Receives images via HTTP, runs quantized model, returns alerts.
"""
import os
import json
import logging
import argparse
import io
import numpy as np
from typing import Dict
from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_file
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install flask opencv-python pillow")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class EdgeInference:
    """Lightweight inference engine for edge devices."""
    
    def __init__(self, model_path: str, model_type: str = "tflite"):
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = (512, 512)
        
        if model_type == "tflite":
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path)
                self.interpreter.allocate_tensors()
                logger.info(f"✓ Loaded TFLite model: {model_path}")
            except ImportError:
                logger.error("tflite_runtime not available")
                self.interpreter = None
        else:
            logger.error(f"Unknown model type: {model_type}")
            self.interpreter = None
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Resize
        image = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Add batch dimension
        image = np.expand_dims(image, 0)
        
        return image.astype(np.float32)
    
    def infer(self, image: np.ndarray) -> Dict:
        """Run inference."""
        
        if self.interpreter is None:
            return {"error": "Model not loaded"}
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Get input/output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        masks = self.interpreter.get_tensor(output_details[0]['index'])
        scores = self.interpreter.get_tensor(output_details[1]['index'])
        
        # Compute severity
        severity = float(np.mean(masks[masks > 0.5])) * 100 if np.any(masks > 0.5) else 0.0
        
        return {
            "severity_pct": severity,
            "mask_shape": masks.shape,
            "confidence": float(np.max(scores))
        }


# Global inference engine
inference_engine = None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@app.route("/infer", methods=["POST"])
def infer():
    """
    Inference endpoint.
    Expects multipart/form-data with 'image' file.
    Returns JSON alert with severity and suggested action.
    """
    
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Read image
        stream = io.BytesIO(file.read())
        image = Image.open(stream).convert("RGB")
        image_np = np.array(image)
        
        # Infer
        result = inference_engine.infer(image_np)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Generate alert
        severity_pct = result["severity_pct"]
        severity_level = "LOW" if severity_pct <= 5 else "MODERATE" if severity_pct <= 20 else "HIGH"
        actions = {"LOW": "monitor", "MODERATE": "targeted_pesticide", "HIGH": "remove_crop_patch"}
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity_pct": severity_pct,
            "severity_level": severity_level,
            "suggested_action": actions[severity_level],
            "confidence": result["confidence"]
        }
        
        logger.info(f"Inference: {severity_pct:.2f}% ({severity_level})")
        
        return jsonify(alert), 200
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """Get server status."""
    return jsonify({
        "status": "running",
        "model_type": inference_engine.model_type if inference_engine else "none",
        "device": "raspberry_pi"  # or jetson_nano, etc
    }), 200


def main():
    parser = argparse.ArgumentParser(description="Edge Inference Server")
    parser.add_argument("--model", type=str, default="./model.tflite",
                       help="Path to model")
    parser.add_argument("--model-type", type=str, default="tflite",
                       choices=["tflite", "onnx"],
                       help="Model format")
    parser.add_argument("--port", type=int, default=5000,
                       help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    global inference_engine
    inference_engine = EdgeInference(args.model, args.model_type)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
