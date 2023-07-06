from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pruebas satisfactorias entre 52% y 92%
# MODEL = tf.keras.models.load_model("saved_models/v3-20")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 55% y 98%
# MODEL = tf.keras.models.load_model("saved_models/v4-25")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 58% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v5-30")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 68% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v6-40")
# CLASS_NAMES = [ "Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 78% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v7-50")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 89% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v8-75")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 91% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v9-100")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

# Pruebas satisfactorias entre 95% y 99%
# MODEL = tf.keras.models.load_model("saved_models/v10-150")
# CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]

MODEL = tf.keras.models.load_model("saved_models/v8-75")
CLASS_NAMES = ["Planta Enferma Trips", "Planta Saludable"]
@app.get("/ping")
async def ping():
    return "Detecctor de Trips"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'Predicción': predicted_class,
        'Confianza': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)