import io
import numpy as np
from fastapi import FastAPI, File
import uvicorn
import PIL
import keras
import utils
import config

app = FastAPI()
model = keras.models.load_model('src/classifier/model.h5')


@app.post("/predict/")
async def create_upload_file(image: bytes = File()):
    image = PIL.Image.open(io.BytesIO(image))
    image = utils.preprocess_image(image)

    model_pred = model.predict(image)
    predicted_class = np.argmax(model_pred)
    prediction = config.CLASSES[predicted_class]

    return {"Predicted class: ": prediction}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
