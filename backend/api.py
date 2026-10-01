from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import sys

# Add the current directory to sys.path to fix module resolution errors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import model_fn, predict_fn

app = FastAPI(title="Video Sentiment Analysis API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to hold the loaded model
app.state.model_dict = None
MODEL_DIR = "./model"
TEMP_DIR = "./temp"

@app.on_event("startup")
async def load_model():
    print("Loading model into memory...")
    try:
        app.state.model_dict = model_fn(MODEL_DIR)
        print("Model loaded successfully.")
        # Ensure temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if app.state.model_dict is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video.")

    # Save uploaded file to a temporary location
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference
        input_data = {'video_path': temp_filepath}
        predictions = predict_fn(input_data, app.state.model_dict)
        
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup the uploaded file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
