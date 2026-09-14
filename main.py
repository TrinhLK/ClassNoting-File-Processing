# main.py
# docker build -t nvhung3001/ai-meeting-assistant:v7 .
# docker build --no-cache -t nvhung3001/ai-meeting-assistant:v7 .
## 2. Push (Dùng lệnh vòng lặp cho chắc ăn)
# $count = 1; while($true) { Write-Host "Lần thử thứ $count..." -ForegroundColor Cyan; docker push nvhung3001/ai-meeting-assistant:v7; if ($?) { Write-Host "✅ XONG!" -ForegroundColor Green; break }; $count++ }
import uuid
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_engine import MeetingAssistant
from pydub import AudioSegment

# --- KHỞI TẠO APP & AI ENGINE ---
app = FastAPI(title="Meeting Assistant API")
  
# Setup CORS cho Next.js (Port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo Engine AI
ai_engine = MeetingAssistant()

# Database giả trong bộ nhớ (Job Queue)
# Cấu trúc: { "job_id": { "status": "...", "type": "...", "result": "...", "error": "..." } }
jobs_db: Dict[str, dict] = {}

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- MODELS ---
class TranscriptionRequest(BaseModel):
    url: str
    language: str = "vi"  # "vi" (default) or "en"

class HybridRequest(BaseModel):
    audio_url: str
    diarization: list
    
# --- BACKGROUND TASKS (WORKERS) ---

def task_transcribe(job_id: str, file_path: Path, language: str = "vi"):
    """Worker xử lý Audio -> Text"""
    try:
        jobs_db[job_id]["status"] = "processing"
        
        # Gọi AI Engine
        segments = ai_engine.process_audio_to_transcript(str(file_path), language=language)
        
        if isinstance(segments, dict) and "error" in segments:
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["error"] = segments["error"]
        else:
            # [UPDATED FOR CLIENT COMPATIBILITY]
            # Client bug: if transcript is [], logic falls through to parseTranscriptFile([]) -> Crash
            # Fix: Return None/Null if empty
            jobs_db[job_id]["output"] = { "transcript": segments if segments else None } 
            jobs_db[job_id]["status"] = "COMPLETED"
            
            print(f"✅ [Job {job_id}] Xử lý xong {len(segments)} segments.")
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        print(f"❌ Error Job {job_id}: {e}")
    finally:
        # Xóa file tạm sau khi xong (hoặc lỗi)
        if file_path.exists():
            os.remove(file_path)

def task_process_hybrid(job_id: str, audio_url: str, diarization: list):
    """Worker xử lý Hybrid Transcription (Async)"""
    try:
        jobs_db[job_id]["status"] = "processing"
        
        # 1. Download & Convert
        url_path = audio_url.split("?")[0]
        ext = Path(url_path).suffix or ".tmp"
        download_path = UPLOAD_DIR / f"{job_id}_hybrid{ext}"
        wav_path = UPLOAD_DIR / f"{job_id}_hybrid.wav"
        
        print(f"⬇️ [Job {job_id}] Downloading Hybrid Audio...")
        urllib.request.urlretrieve(audio_url, download_path)
        
        print(f"🔄 [Job {job_id}] Converting to WAV 16kHz...")
        audio = AudioSegment.from_file(str(download_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(wav_path), format="wav")
        
        if download_path.exists(): os.remove(download_path)

        # 2. Process AI (Zipformer + Alignment)
        print(f"🧠 [Job {job_id}] Running AI Alignment...")
        segments = ai_engine.process_hybrid_transcription(str(wav_path), diarization)
        
        if wav_path.exists(): os.remove(wav_path)
        
        if isinstance(segments, dict) and "error" in segments:
            raise Exception(segments["error"])

        # 3. Save Result
        # Output format matching what PollingManager expects
        jobs_db[job_id]["output"] = { "transcript": segments }
        jobs_db[job_id]["status"] = "COMPLETED"
        print(f"✅ [Job {job_id}] Hybrid Process Completed: {len(segments)} segments.")

    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        print(f"❌ Error Job {job_id}: {e}")
        if 'wav_path' in locals() and wav_path.exists(): os.remove(wav_path)


@app.get("/")
def health_check():
    return {"status": "online", "backend": "Meeting Assistant AI"}

# 1. API Upload Audio
@app.post("/api/v1/transcribe")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), language: str = "vi"):
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    # Lưu file từ request xuống ổ cứng
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Tạo Job
    jobs_db[job_id] = {"status": "queued", "type": "transcribe"}
    
    # Đẩy việc cho Worker chạy ngầm
    background_tasks.add_task(task_transcribe, job_id, file_path, language)
    
    return {"job_id": job_id, "message": "Audio received. Processing started."}

@app.post("/api/v1/transcribe_url")
async def transcribe_from_url(background_tasks: BackgroundTasks, req: TranscriptionRequest):
    job_id = str(uuid.uuid4())
    
    # [FIX] Logic tải và convert thông minh
    # 1. Đoán đuôi file gốc từ URL (để pydub/ffmpeg nhận diện đúng)
    url_path = req.url.split("?")[0] # Bỏ query params
    ext = Path(url_path).suffix or ".tmp"
    
    # 2. Đường dẫn tạm
    download_path = UPLOAD_DIR / f"{job_id}_raw{ext}"
    wav_path = UPLOAD_DIR / f"{job_id}_converted.wav"
    
    try:
        print(f"Downloading from {req.url}...")
        # Tải file gốc
        urllib.request.urlretrieve(req.url, download_path)
        
        # 3. Convert sang WAV 16kHz Mono (Chuẩn cho Zipformer/Pyannote)
        print("Converting to WAV 16kHz...")
        audio = AudioSegment.from_file(str(download_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(wav_path), format="wav")
        
        # Xóa file gốc cho nhẹ
        if download_path.exists():
            os.remove(download_path)
            
    except Exception as e:
         if download_path.exists(): os.remove(download_path)
         print(f"❌ Download/Convert Error: {e}")
         raise HTTPException(status_code=400, detail=f"Error processing audio URL: {str(e)}")

    # Tạo Job
    jobs_db[job_id] = {"status": "queued", "type": "transcribe"}
    
    # Đẩy việc cho Worker (Dùng file WAV đã convert)
    background_tasks.add_task(task_transcribe, job_id, wav_path, req.language)
    
    return {"job_id": job_id, "message": "Audio URL received. Downloading & Processing."}

# 3. API Check Trạng thái Job (Polling)
@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/transcribe_hybrid")
async def transcribe_hybrid(background_tasks: BackgroundTasks, req: HybridRequest):
    """
    Endpoint mới cho Hybrid Pipeline (Async)
    Input: Audio URL + Bot Diarization
    Output: Job ID
    """
    job_id = str(uuid.uuid4())
    
    # Tạo Job
    jobs_db[job_id] = {
        "status": "queued", 
        "type": "hybrid_transcription",
        "created_at": str(job_id)
    }
    
    # Đẩy việc cho Worker
    background_tasks.add_task(task_process_hybrid, job_id, req.audio_url, req.diarization)
    
    return {"job_id": job_id, "message": "Hybrid processing started.", "status": "queued"}