import tensorflow as tf
import numpy as np
import base64
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)


@app.get("/")
def home():
    return {"message": "Backend is working!"}


model = tf.keras.models.load_model("analyzer_model.keras")

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/predict")
async def predict_skin_type(file: UploadFile = File(...)):  
    try:
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

       
        processed_image = preprocess_image(image)

        
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)

        classes = ["N", "O", "R"]
        result = classes[predicted_class]

        return {"skin_type": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

