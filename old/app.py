
# To run this application, you will need to install the following libraries:
# pip install fastapi uvicorn python-multipart sounddevice scipy joblib librosa

import fastapi
import sounddevice as sd
import joblib
import numpy as np
import librosa

app = fastapi.FastAPI()

# Load the trained model
model = joblib.load("random_forest_model.joblib")

def record_audio(duration=1, fs=8000):
    """Records audio from the microphone."""
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return recording.flatten(), fs

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return fastapi.responses.Response(status_code=204)

@app.post("/inference")
async def perform_inference():
    """Records audio and performs inference."""
    try:
        # Record audio
        audio_data, fs = record_audio()

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=fs, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        features = mfccs_processed.reshape(1, -1)

        # Perform inference
        prediction = model.predict(features)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return fastapi.Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
