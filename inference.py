import tensorflow as tf
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
import time

# Setup logger
logger.add("app.log", rotation="1 day", level="INFO")

# Create FastAPI instance
app = FastAPI()

# Load inference model
try:
    loaded_model = tf.keras.layers.TFSMLayer("saved_model_trashlab", call_endpoint="serving_default")
    logger.info("Model loaded successfully for inference.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

def preprocess_image_bytes(image_bytes: bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to read image.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        raise

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Request from {request.client.host} - Upload file: {file.filename}")

    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file format request")
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        img = preprocess_image_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        prediction = loaded_model(img)

        if isinstance(prediction, dict):
            prediction = list(prediction.values())[0]
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()

        label = "Organik" if prediction[0][0] < 0.5 else "Anorganik"
        confidence = 1 - prediction[0][0] if label == "Organik" else prediction[0][0]

        response_time = time.time() - start_time
        logger.info(f"Prediction: {label}, Confidence: {confidence:.4f}, Response Time: {response_time:.4f} seconds")

        return JSONResponse(content={
            "label": label,
            "confidence": float(confidence),
            "response_time": response_time
        })

    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during prediction.")
