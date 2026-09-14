import runpod
import os
import re
import urllib.request
import traceback  # <--- [QUAN TRỌNG] Thư viện để in chi tiết lỗi
import subprocess
from urllib.parse import urlparse
from ai_engine import MeetingAssistant

# ==========================================
# HẰNG SỐ & HELPER
# ==========================================
# Allowlist các đuôi audio được chấp nhận (lowercase, có dấu chấm).
# Chặn path-traversal & chặn file nguy hiểm (.exe, .sh, ...).
_AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus",
    ".webm", ".mp4", ".aac", ".wma", ".oga", ".mka",
}
_FFMPEG_STDERR_TAIL_LINES = 20


def _safe_ext_from_url(url: str, default: str = ".mp3") -> str:
    """
    Lấy đuôi file từ URL an toàn.
    - Bỏ query string & fragment để tránh lấy nhầm '?alt=media&token=...' làm ext.
    - Chỉ chấp nhận ext nằm trong allowlist audio để chặn file nguy hiểm (.exe, .sh...).
    """
    try:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext in _AUDIO_EXTS:
            return ext
        return default
    except Exception:
        return default


def _run_ffmpeg(input_path: str, output_path: str) -> None:
    """
    Chạy ffmpeg convert audio. Nếu fail, log tail stderr để dễ debug
    (trước đây nuốt hết stderr bằng DEVNULL nên không biết lý do).
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", output_path,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-_FFMPEG_STDERR_TAIL_LINES:])
        raise RuntimeError(
            f"ffmpeg exit {proc.returncode}.\n"
            f"Input: {input_path}\nOutput: {output_path}\n"
            f"stderr (last {_FFMPEG_STDERR_TAIL_LINES} lines):\n{stderr_tail}"
        )

# ==========================================
# 1. KHỞI TẠO GLOBAL (CÓ BẢO VỆ)
# ==========================================
ai_engine = None

print("🚀 [INIT] Đang khởi động AI Engine...")
try:
    # Bọc việc khởi tạo trong Try/Except để nếu lỗi OOM/Driver thì còn biết
    ai_engine = MeetingAssistant()
    print("✅ [INIT] AI Engine sẵn sàng!")
except Exception as e:
    print("❌ [INIT ERROR] LỖI KHỞI TẠO NGHIÊM TRỌNG:")
    # In toàn bộ đường dẫn lỗi ra Log của RunPod
    print(traceback.format_exc()) 

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH
# ==========================================
def download_file(url, filename):
    try:
        print(f"⬇️ Đang tải file từ: {url}")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Đã tải xong: {filename}")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải file: {e}")
        return False

# ==========================================
# 3. HANDLER CHÍNH
# ==========================================
def handler(job):
    """
    Input event từ RunPod:
    {
        "input": {
            "action": "transcribe" | "transcribe_hybrid",
            "audio_url": "...", 
            "text": "..."
        }
    }
    """
    # Kiểm tra xem Engine có sống không trước khi nhận việc
    if ai_engine is None:
        return {
            "status": "failed", 
            "error": "AI Engine khởi động thất bại. Hãy xem System Logs trên RunPod."
        }

    job_input = job["input"]
    job_id = job["id"]
    action = job_input.get("action", "transcribe")
    language = job_input.get("language", "vi")  # "vi" (default) or "en"
    
    print(f"📥 [JOB {job_id}] Nhận yêu cầu: {action} | language={language}")

    try:
        # --- XỬ LÝ TRANSCRIBE ---
        if action == "transcribe":
            audio_url = job_input.get("audio_url")

            # 1. Lấy đuôi file gốc từ URL (an toàn — bỏ query string)
            ext = _safe_ext_from_url(audio_url)
            print(f"   Detected extension: {ext}")

            # Tên file tải về (giữ nguyên đuôi gốc để tránh lỗi header)
            original_filename = f"/tmp/{job_id}_original{ext}"
            # Tên file WAV để đưa vào xử lý (An toàn nhất cho soundfile)
            wav_filename = f"/tmp/{job_id}_audio.wav"

            if not download_file(audio_url, original_filename):
                return {"status": "failed", "error": "Cannot download audio file"}

            try:
                # 2. Convert sang WAV bằng ffmpeg
                print(f"🔄 Converting {original_filename} -> {wav_filename}...")
                _run_ffmpeg(original_filename, wav_filename)

                # 3. Gọi AI Engine với file WAV chuẩn
                result = ai_engine.process_audio_to_transcript(wav_filename, language=language)
                return {"status": "success", "transcript": result}

            except Exception as e:
                print(f"❌ Transcribe Error: {e}")
                print(traceback.format_exc())
                raise e # Ném lỗi ra để catch bên ngoài log
            finally:
                # Dọn dẹp cả 2 file
                if os.path.exists(original_filename): os.remove(original_filename)
                if os.path.exists(wav_filename): os.remove(wav_filename)
                print(f"🧹 Dọn dẹp file tạm xong.")

        # --- XỬ LÝ HYBRID TRANSCRIPTION ---
        elif action == "transcribe_hybrid":
            audio_url = job_input.get("audio_url")
            diarization = job_input.get("diarization") or []

            if not audio_url: return {"status": "failed", "error": "Missing audio_url"}

            # 1. Reuse logic download & convert (ext an toàn — bỏ query string)
            ext = _safe_ext_from_url(audio_url)
            print(f"   Detected extension: {ext}")
            original_filename = f"/tmp/{job_id}_hybrid_original{ext}"
            wav_filename = f"/tmp/{job_id}_hybrid.wav"

            try:
                if not download_file(audio_url, original_filename):
                    return {"status": "failed", "error": "Cannot download audio file"}

                print(f"🔄 Converting {original_filename} -> {wav_filename}...")
                _run_ffmpeg(original_filename, wav_filename)

                # 2. Call Hybrid Engine
                segments = ai_engine.process_hybrid_transcription(wav_filename, diarization)

                if isinstance(segments, dict) and "error" in segments:
                     return {"status": "failed", "error": segments["error"]}

                return {"status": "success", "segments": segments}

            finally:
                if os.path.exists(original_filename): os.remove(original_filename)
                if os.path.exists(wav_filename): os.remove(wav_filename)

        else:
            return {"status": "failed", "error": f"Unknown action: {action}"}

    except Exception as e:
        # [QUAN TRỌNG] Bắt lỗi Runtime (lỗi khi đang xử lý)
        print(f"❌ [JOB {job_id}] LỖI XỬ LÝ (RUNTIME ERROR):")
        print(traceback.format_exc()) # In lỗi chi tiết ra RunPod Logs
        
        # Trả về lỗi cho Client biết luôn
        return {
            "status": "failed", 
            "error": str(e),
            "traceback": traceback.format_exc() # (Tùy chọn) Gửi kèm trace cho Client debug
        }

# Bắt đầu lắng nghe request
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})