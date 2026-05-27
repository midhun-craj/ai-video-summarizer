from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from models import process_video

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Audio-Video Summarizer API is running."}

@app.post("/summarize-video")
async def summarize_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        return JSONResponse(status_code=400, content={"error": "Unsupported file format."})

    video_bytes = await file.read()
    try:
        summarized_video_bytes = process_video(video_bytes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return StreamingResponse(
        io.BytesIO(summarized_video_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=summarized_{file.filename}"}
    )


    # To run the app use the "uvicorn main:app --reload --port 8001"