import librosa
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("457model.keras")

# same settings as training
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

@app.post("/predict")
async def predict(file: UploadFile):
    # load audio
    audio, sr = librosa.load(file.file, sr=SAMPLE_RATE)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # match training input shape
    mel_db = np.expand_dims(mel_db, axis=(0, -1))  # shape (1, time, n_mels, 1)

    # run model
    prediction = model.predict(mel_db)

    return {"prediction": prediction.tolist()}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)