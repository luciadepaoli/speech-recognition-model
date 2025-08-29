from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import librosa
import joblib

app = FastAPI()

# Load trained RandomForest
clf = joblib.load("random_forest_model.joblib")

# Parameters
n_mfcc = 13
SR = 8000

def extract_mfcc_from_array(audio_array):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=SR, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return mfcc_mean

# Serve page with recording
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head><title>Digit Recorder</title></head>
    <body>
        <h2>Record a digit (0-9)</h2>
        <button onclick="startRecording()">Record</button>
        <p id="status"></p>
        <script>
        let mediaRecorder;
        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                let chunks = [];
                mediaRecorder.ondataavailable = e => chunks.push(e.data);
                mediaRecorder.onstop = e => {
                    // Convert audio to Float32 array
                    let reader = new FileReader();
                    reader.onload = function() {
                        let arrayBuffer = reader.result;
                        let audioContext = new AudioContext();
                        audioContext.decodeAudioData(arrayBuffer, buffer => {
                            let data = buffer.getChannelData(0);
                            fetch("/predict", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify(Array.from(data))
                            })
                            .then(resp => resp.json())
                            .then(data => document.getElementById("status").innerText = "Predicted digit: " + data.prediction);
                        });
                    };
                    reader.readAsArrayBuffer(chunks[0]);
                };
                mediaRecorder.start();
                setTimeout(() => mediaRecorder.stop(), 500); // 0.5s
            });
        }
        </script>
    </body>
    </html>
    """

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request):
    audio_list = await request.json()
    audio_array = np.array(audio_list, dtype=np.float32)
    mfcc_features = extract_mfcc_from_array(audio_array)
    pred = clf.predict(mfcc_features)[0]
    return JSONResponse({"prediction": int(pred)})
