# Server-local — Meeting Assistant AI

Server xử lý audio cho **AI Meeting Assistant**: nhận file ghi âm cuộc họp và trả về transcript có gắn nhãn người nói (speaker-aware transcription).

**Pipeline:** `Sherpa-ONNX Zipformer` (ASR) + `Pyannote Audio` (diarization) + `FFmpeg` (preprocessing).

Hỗ trợ 2 chế độ triển khai song song:
- **RunPod Serverless** (production) — `handler.py`
- **FastAPI local** (dev/test) — `main.py`

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chạy local (FastAPI)](#-chạy-local-fastapi)
- [Build & deploy RunPod](#-build--deploy-runpod)
- [API Endpoints](#-api-endpoints)
- [Kết nối từ client](#-kết-nối-từ-client)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Biến môi trường](#-biến-môi-trường)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Tính năng

| Endpoint | Mô tả |
|---|---|
| `transcribe` | Upload audio → transcript có speaker labels (Pyannote tự tách người nói) |
| `transcribe_url` | Nhận URL audio (Firebase/HTTPS) → transcript có speaker labels |
| `transcribe_hybrid` | Nhận URL + diarization từ bot bên ngoài → ASR + gán speaker từ bot |
| `transcribe_url` (en) | Hỗ trợ tiếng Anh (`language="en"`) qua Zipformer GigaSpeech |

**Tính năng xử lý:**
- Auto-convert audio sang WAV 16kHz mono (chuẩn cho Zipformer/Pyannote)
- VAD (Silero) cắt đoạn im lặng trước khi ASR → tiết kiệm RAM và tăng tốc
- Smoothing diarization: gộp các đoạn ngắt quãng nhỏ (`A → B(0.3s) → A` → gộp thành `A`)
- Hỗ trợ file dài >5 giờ (buffer VAD 18000s)
- Token merging thông minh cho tiếng Việt (phonotactic rules)
- Fallback: nếu VAD phát hiện quá ít giọng → chunk 30s cứng

---

## 🛠 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| OS | Linux (Ubuntu 22.04+ khuyến nghị) / Windows 10+ / macOS |
| Python | 3.11+ |
| RAM | ≥ 8 GB (16 GB khuyến nghị cho audio dài) |
| VRAM | ≥ 6 GB (chạy Pyannote trên GPU) |
| Disk | ~10 GB cho Docker image + cache models |
| ffmpeg | Bắt buộc (Dockerfile tự cài) |
| HuggingFace Token | Bắt buộc, để tải Pyannote diarization model |

---

## 📦 Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/NguyenVanHung2004/Server-local-ai-meeting-assistant.git
cd Server-local-ai-meeting-assistant
```

### 2. (Khuyến nghị) Tạo virtual environment

```bash
python -m venv env
# Linux / macOS
source env/bin/activate
# Windows
env\Scripts\activate
```

### 3. Cài dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Cài `pyannote-audio` 4.0.2 (workaround dependency conflict)

Vì `pyannote.audio` 4.x có dependency conflict với torch version mới, cần cài theo 2 bước:

```bash
# Cài pyannote-audio KHÔNG kèm dependencies
pip install --no-deps pyannote-audio==4.0.2

# Cài bù các package phụ cần thiết
pip install \
    "lightning>=2.0.0" \
    "pyannote-core>=6.0.1" \
    "pyannote-pipeline>=4.0.0" \
    "pyannote-database>=6.1.0" \
    "pyannote-metrics>=4.0.0" \
    "torch-audiomentations>=0.12.0" \
    "torchmetrics>=1.8.2" \
    "asteroid-filterbanks>=4.0.0" \
    "einops>=0.8.0" \
    "speechbrain>=1.0.0" \
    "semver>=3.0.0" \
    "omegaconf>=2.1" \
    "docopt>=0.6.2" \
    "opentelemetry-api>=1.0.0" \
    "opentelemetry-sdk>=1.0.0" \
    "opentelemetry-exporter-otlp>=1.0.0" \
    "tensorboard>=2.0.0" \
    "soundfile>=0.12.1" \
    "rich>=12.0.0"
```

### 5. Cài `ffmpeg`

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 6. Set HuggingFace Token

Tạo token tại https://huggingface.co/settings/tokens và accept terms tại https://huggingface.co/pyannote/speaker-diarization-community-1.

**Cách khuyến nghị:** Copy `.env.example` → `.env` và điền giá trị:

```bash
cp .env.example .env
# Sửa file .env, thay hf_your_token_here bằng token thật
```

**Cách khác** (nếu không dùng .env):

```bash
# Linux / macOS
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
setx HF_TOKEN "hf_your_token_here"
```

> 📄 Xem [`.env.example`](.env.example) để biết tất cả biến môi trường được hỗ trợ.

### 7. Tải models (chạy 1 lần)

```bash
python builder.py
```

Script sẽ tải về `/workspace/cache` (hoặc `./cache` nếu local):
- Pyannote diarization model (~50 MB)
- Vietnamese Zipformer ASR + Silero VAD (~280 MB)
- English Zipformer ASR + Silero VAD (~290 MB)

---

## 🧪 Chạy local (FastAPI) — **CHỈ ĐỂ TEST**

> ⚠️ **Server này KHÔNG dành cho production.** Chỉ dùng để dev/test trên máy local.
> Môi trường production chính là **RunPod Serverless** — xem mục ngay bên dưới.

Khởi động server FastAPI trên port 8000:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- CORS đã mở sẵn cho `http://localhost:3000` (Next.js client)
- Job queue là **in-memory**, mất khi restart server
- KHÔNG scale được, KHÔNG có auto-retry, KHÔNG có persistent storage

> 💡 **Khuyến nghị**: Skip mục này, đi thẳng đến phần [Build & deploy RunPod](#-build--deploy-runpod-production) bên dưới.

---

## 🐳 Build & deploy RunPod (Production)

> 🎯 **Đây là cách deploy chính thức.** Server sẽ chạy trên GPU cloud của RunPod, scale tự động theo lưu lượng, có persistent cache qua Network Volume.

Có **2 cách** build Docker image — khuyến nghị dùng cách 1 (build trực tiếp từ Git).

---

### ✅ Cách 1: Build từ GitHub (Khuyến nghị — không cần Docker Hub)

RunPod hỗ trợ tự động clone repo và build image. Mỗi lần push code mới lên GitHub, RunPod sẽ tự rebuild.

1. Vào https://www.runpod.io/console/serverless
2. **New Endpoint** → ở mục **Container Image**, chọn **"GitHub Repo"** thay vì Docker Image
3. Điền thông tin:
   - **GitHub URL**: `https://github.com/NguyenVanHung2004/Server-local-ai-meeting-assistant`
   - **Branch**: `main` (hoặc `cleanup/remove-unused-files` để dùng bản đã dọn rác)
   - **Dockerfile path**: `Dockerfile` (mặc định)
   - **Build context**: `.` (mặc định)
4. Cấu hình **Environment Variables:**
   - `HF_TOKEN` = `hf_your_token_here`
   - `HF_HOME` = `/workspace/cache`
5. **GPU:** RTX 3090 / A10G / A100 (≥ 12 GB VRAM khuyến nghị)
6. **Container Disk:** ≥ 20 GB (chứa models)
7. **Volume Disk:** ≥ 30 GB (cache)
8. **Max Workers:** 1-3 (tùy quota)
9. Bấm **Deploy** → RunPod sẽ tự clone repo + `docker build` + khởi động endpoint

> 💡 Mỗi lần push code mới, RunPod tự rebuild lại image. Nếu muốn trigger build thủ công, vào **Endpoint → Build → Rebuild**.

---

### Cách 2: Build từ local + push lên Docker Hub (legacy)

Dùng khi bạn muốn kiểm soát Docker image chặt hơn hoặc dùng image ở nơi khác (không phải RunPod).

**1. Build image local:**
```bash
docker build -t nvhung3001/ai-meeting-assistant:v7 .
```

**2. Push lên Docker Hub:**
```bash
docker push nvhung3001/ai-meeting-assistant:v7
```

**3. Tạo RunPod Serverless Endpoint:**
1. Vào https://www.runpod.io/console/serverless
2. **New Endpoint** → chọn **"Docker Image"** thay vì GitHub Repo
3. Image name: `nvhung3001/ai-meeting-assistant:v7`
4. Cấu hình Environment Variables + GPU + Disk như Cách 1
5. Lấy **Endpoint ID** và **API Key**

> ⚠️ Cách này phải rebuild + push thủ công mỗi khi code đổi.

---

## 📡 API Endpoints

Server cung cấp **2 giao diện**:
- **Production**: RunPod Serverless (`handler.py`) — scale tự động, GPU cloud
- **Dev/Test**: FastAPI local (`main.py`) — chỉ chạy trên máy dev

> 🎯 **Client (Next.js) chỉ cần quan tâm RunPod API.** FastAPI local chỉ dành cho test.

---

### A. RunPod Serverless API (Production) — **DÙNG CÁI NÀY**

RunPod Serverless cung cấp 4 endpoints HTTP chuẩn để tương tác với worker:

| Method | URL | Mục đích |
|---|---|---|
| `POST` | `/v2/{ENDPOINT_ID}/run` | Submit job bất đồng bộ (trả về `id` ngay) |
| `GET` | `/v2/{ENDPOINT_ID}/status/{job_id}` | Poll trạng thái + lấy output |
| `POST` | `/v2/{ENDPOINT_ID}/cancel/{job_id}` | Hủy job đang chờ/running |
| `GET` | `/v2/{ENDPOINT_ID}/health` | Kiểm tra endpoint có worker sẵn không |

**Base URL:** `https://api.runpod.ai`

**Auth header (mọi request):**
```
Authorization: Bearer {RUNPOD_API_KEY}
Content-Type: application/json
```

#### Action mà server hỗ trợ (qua field `input.action`)

| Action | Mô tả | Async? |
|---|---|---|
| `transcribe` | Transcribe audio từ URL — server tự chạy Pyannote để tách speaker | ✅ Nên async |
| `transcribe_hybrid` | Transcribe audio + dùng diarization từ bot bên ngoài | ✅ Nên async |

> Chọn `action` trong body request. Các field khác phụ thuộc vào action.

---

### 1. Kiểm tra endpoint health (RunPod)

**Request:**
```http
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/health
Authorization: Bearer {RUNPOD_API_KEY}
```

**Response 200 (endpoint sẵn sàng):**
```json
{
  "jobs": {
    "completed": 42,
    "failed": 1,
    "inProgress": 2,
    "inQueue": 0,
    "retried": 0,
    "scheduled": 0
  },
  "workers": {
    "idle": 1,
    "running": 1,
    "throttled": 0
  }
}
```

**Response 503 (không có worker sẵn — endpoint đang scale up):**
```json
{ "error": "No workers available" }
```

> Worker có thể mất 30-60s để cold-start. Nên check health trước khi submit job quan trọng.

---

### 2. Submit job async (RunPod `/run`)

Đây là cách **khuyến nghị** cho audio dài (>1 phút). Server trả về `id` ngay lập tức, không block.

**Request:**
```http
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer {RUNPOD_API_KEY}
Content-Type: application/json

{
  "input": {
    "action": "transcribe",
    "audio_url": "https://storage.googleapis.com/.../meeting.mp3",
    "language": "vi"
  }
}
```

**Body parameters:**

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `input.action` | ✅ | `"transcribe"` hoặc `"transcribe_hybrid"` |
| `input.audio_url` | ✅ | URL công khai (HTTPS) tới file audio (mp3, wav, m4a, webm, ...) |
| `input.language` | ❌ | `"vi"` (mặc định) hoặc `"en"` |
| `input.diarization` | ⚠️ | Chỉ với `transcribe_hybrid`: list các segment `{start_time, end_time, speaker}` |

**Response 200 (job đã vào queue):**
```json
{ "id": "01HXYZ...RUNPOD-JOB-ID", "status": "IN_QUEUE" }
```

**Status codes:**
- `IN_QUEUE` — đang chờ worker
- `IN_PROGRESS` — worker đang xử lý
- `COMPLETED` — xong, có `output`
- `FAILED` — lỗi, có `error` trong `output`
- `CANCELLED` — đã hủy
- `TIMED_OUT` — quá thời gian (timeout mặc định 600s, có thể tăng)

---

### 3. Poll job status (RunPod `/status`)

Sau khi submit, gọi liên tục để lấy output:

**Request:**
```http
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}
Authorization: Bearer {RUNPOD_API_KEY}
```

**Response khi `COMPLETED`:**
```json
{
  "id": "01HXYZ...",
  "status": "COMPLETED",
  "createdAt": "2026-08-06T10:00:00.000Z",
  "delayTime": 1234,
  "executionTime": 4521,
  "output": {
    "status": "success",
    "transcript": [
      {
        "id": "0",
        "speakerId": "SPEAKER_00",
        "text": "Chào mừng quý vị đến với buổi họp.",
        "start": 0.5,
        "end": 3.2,
        "words": [
          { "word": "chào", "start": 0.5, "end": 0.8, "speaker": "SPEAKER_00" },
          { "word": "mừng", "start": 0.9, "end": 1.1, "speaker": "SPEAKER_00" }
        ]
      },
      {
        "id": "1",
        "speakerId": "SPEAKER_01",
        "text": "Cảm ơn anh đã giới thiệu.",
        "start": 4.0,
        "end": 6.5,
        "words": [ /* ... */ ]
      }
    ]
  }
}
```

**Response khi `FAILED`:**
```json
{
  "id": "01HXYZ...",
  "status": "FAILED",
  "output": {
    "status": "failed",
    "error": "Cannot download audio file",
    "traceback": "Traceback (most recent call last):\n  ..."
  }
}
```

**Field quan trọng trong response:**

| Field | Ý nghĩa |
|---|---|
| `delayTime` | Thời gian chờ trong queue (ms) |
| `executionTime` | Thời gian worker xử lý (ms) |
| `output.status` | `"success"` hoặc `"failed"` |
| `output.transcript[]` | List các segment (mỗi segment = 1 lượt nói của 1 người) |
| `output.error` | Chỉ có khi lỗi |

---

### 4. Hybrid action (RunPod)

Dùng khi **đã có diarization từ bot bên ngoài** (vd: MeetingBaaS, Daily.co bot, Recall.ai). Server chỉ làm ASR và gán speaker theo diarization có sẵn → chính xác hơn Pyannote.

**Request:**
```http
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer {RUNPOD_API_KEY}
Content-Type: application/json

{
  "input": {
    "action": "transcribe_hybrid",
    "audio_url": "https://storage.googleapis.com/.../meeting.mp3",
    "diarization": [
      { "start_time": 0.0,  "end_time": 5.2,  "speaker": "Alice" },
      { "start_time": 5.2,  "end_time": 10.8, "speaker": "Bob" },
      { "start_time": 10.8, "end_time": 18.4, "speaker": "Alice" }
    ]
  }
}
```

**Field `input.diarization[]`:**

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `start_time` (hoặc `start`) | ✅ | Thời điểm bắt đầu (giây, float) |
| `end_time` (hoặc `end`) | ✅ | Thời điểm kết thúc (giây, float) |
| `speaker` | ❌ | Tên người nói. Mặc định `"SPEAKER_00"` |

> Server chấp nhận cả `start_time/end_time` lẫn `start/end`. Mỗi item cũng có thể là JSON string (server tự parse).

**Output `output.segments[]`** (khác với `transcribe` ở key name):
```json
{
  "output": {
    "status": "success",
    "segments": [
      {
        "id": "0",
        "speakerId": "Alice",
        "text": "...",
        "start": 0.5,
        "end": 5.0,
        "words": [ /* ... */ ]
      }
    ]
  }
}
```

> ⚠️ Lưu ý: Hybrid trả về `segments` (không phải `transcript`). Client cần handle cả 2 key.

---

### 5. Cancel job (RunPod `/cancel`)

**Request:**
```http
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/cancel/{job_id}
Authorization: Bearer {RUNPOD_API_KEY}
```

**Response 200:**
```json
{ "id": "01HXYZ...", "status": "CANCELLED" }
```

> Chỉ cancel được job đang `IN_QUEUE`. Job đang `IN_PROGRESS` không cancel được — phải đợi timeout.

---

### 6. Output format — Schema chung

Cả 2 action đều trả về **segment-level** JSON. Mỗi segment = 1 lượt nói liên tục của 1 người.

**Segment object:**
```typescript
type Segment = {
  id: string;            // "0", "1", "2"...
  speakerId: string;     // "SPEAKER_00", "Alice", hoặc tên từ bot
  text: string;          // Câu đầy đủ
  start: number;         // Giây (float)
  end: number;           // Giây (float)
  words: Word[];         // Word-level timestamps
};

type Word = {
  word: string;
  start: number;
  end: number;
  speaker: string;
};
```

**Lưu ý quan trọng:**

- `transcript` có thể là `null` (nếu audio rỗng / không có giọng nói). Client PHẢI handle null.
- Word-level chỉ có khi ASR chạy thành công. Một số trường hợp fallback có thể trả về mảng rỗng.
- `speakerId` mặc định là `SPEAKER_00`, `SPEAKER_01`, ... cho `transcribe`; là tên thật (Alice, Bob, ...) cho `transcribe_hybrid`.

---

## 🧪 FastAPI local (CHỈ ĐỂ TEST — KHÔNG DÙNG CHO PRODUCTION)

> ⚠️ **Skip mục này nếu bạn chỉ cần deploy.** FastAPI local chỉ phục vụ debug trên máy dev.

Khởi động:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/transcribe` | Upload file audio (multipart) |
| `POST` | `/api/v1/transcribe_url` | Transcribe từ URL (JSON) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status (in-memory) |
| `POST` | `/transcribe_hybrid` | Hybrid pipeline (URL + bot diarization) |

### Upload audio file (FastAPI)

**Request:**
```http
POST /api/v1/transcribe
Content-Type: multipart/form-data

file: <binary audio file (mp3/wav/m4a/...)>
language: vi   # hoặc "en"
```

**Response 202:**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "message": "Audio received. Processing started."
}
```

**Sau đó polling:**

```http
GET /api/v1/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Response khi hoàn thành:**
```json
{
  "status": "COMPLETED",
  "type": "transcribe",
  "output": {
    "transcript": [
      {
        "id": "0",
        "speakerId": "SPEAKER_00",
        "text": "Chào mừng quý vị đến với buổi họp.",
        "start": 0.5,
        "end": 3.2,
        "words": [
          { "word": "chào",      "start": 0.5,  "end": 0.8,  "speaker": "SPEAKER_00" },
          { "word": "mừng",      "start": 0.9,  "end": 1.1,  "speaker": "SPEAKER_00" },
          { "word": "quý",       "start": 1.2,  "end": 1.4,  "speaker": "SPEAKER_00" },
          { "word": "vị",        "start": 1.4,  "end": 1.5,  "speaker": "SPEAKER_00" },
          { "word": "đến",       "start": 1.6,  "end": 1.8,  "speaker": "SPEAKER_00" },
          { "word": "với",       "start": 1.8,  "end": 2.0,  "speaker": "SPEAKER_00" },
          { "word": "buổi",      "start": 2.1,  "end": 2.4,  "speaker": "SPEAKER_00" },
          { "word": "họp",       "start": 2.5,  "end": 3.2,  "speaker": "SPEAKER_00" }
        ]
      },
      {
        "id": "1",
        "speakerId": "SPEAKER_01",
        "text": "Cảm ơn anh đã giới thiệu.",
        "start": 4.0,
        "end": 6.5,
        "words": [ ... ]
      }
    ]
  }
}
```

**Response khi lỗi:**
```json
{
  "status": "failed",
  "type": "transcribe",
  "error": "Cannot read audio file"
}
```

> Tương tự cho `/api/v1/transcribe_url` (JSON body `{url, language}`) và `/transcribe_hybrid` (JSON body `{audio_url, diarization[]}`). Polling qua `/api/v1/jobs/{job_id}`. Output JSON có cùng schema với RunPod (xem mục "Output format — Schema chung" ở trên).

---

## 🔌 Kết nối từ client

### Từ Next.js / React

```typescript
// app/lib/api.ts

const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID;

export const startTranscriptionJob = async (
  audioUrl: string,
  language: "vi" | "en" = "vi"
): Promise<string> => {
  const response = await fetch(
    `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({
        input: {
          action: "transcribe",
          audio_url: audioUrl,
          language: language,
        },
      }),
    }
  );
  const data = await response.json();
  if (data.id) return data.id;
  throw new Error("RunPod Error: " + JSON.stringify(data));
};

export const checkJobStatusOnce = async (jobId: string) => {
  const statusUrl = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`;
  const response = await fetch(statusUrl, {
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
  });
  return await response.json();
};
```

### Từ cURL (test local FastAPI)

```bash
# 1. Upload file
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@./meeting.mp3" \
  -F "language=vi" | jq -r .job_id)

# 2. Poll status
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r .status)
  echo "Status: $STATUS"
  if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 3
done

# 3. Lấy kết quả
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq .
```

### Từ cURL (test RunPod)

```bash
# 1. Submit job
RESPONSE=$(curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "transcribe",
      "audio_url": "https://example.com/meeting.mp3",
      "language": "vi"
    }
  }')

JOB_ID=$(echo $RESPONSE | jq -r .id)
echo "Submitted job: $JOB_ID"

# 2. Poll status
while true; do
  STATUS=$(curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
    -H "Authorization: Bearer $RUNPOD_API_KEY" | jq -r .status)
  echo "Status: $STATUS"
  if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ]; then break; fi
  sleep 5
done

# 3. Lấy output
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq .output
```

---

## 📂 Cấu trúc dự án

```
Server-local/
├── ai_engine.py             # Core: MeetingAssistant class (Zipformer + Pyannote)
├── builder.py               # Script tải model lúc Docker build
├── check_versions.py        # Utility: in version các package đã cài
├── handler.py               # RunPod Serverless handler
├── main.py                  # FastAPI server local (port 8000)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image cho RunPod
├── .gitignore               # Ignore model_vi/, model_en/, env/, temp_uploads/
├── versions_local.txt       # Snapshot pip list (local env)
│
├── model_vi/                # Vietnamese Zipformer (epoch-20, ~280MB)
│   ├── encoder-epoch-20-avg-10.onnx
│   ├── decoder-epoch-20-avg-10.onnx
│   ├── joiner-epoch-20-avg-10.onnx
│   ├── silero_vad.onnx      # Voice Activity Detection
│   └── tokens.txt           # BPE tokens
│
└── model_en/                # English Zipformer GigaSpeech (~290MB)
    ├── encoder-epoch-30-avg-1.onnx
    ├── decoder-epoch-30-avg-1.onnx
    ├── joiner-epoch-30-avg-1.onnx
    ├── silero_vad.onnx
    └── tokens.txt
```

> Thư mục `model_vi/` và `model_en/` được `.gitignore` (quá lớn để commit). Chạy `python builder.py` để tự động tải về.

---

## 🔐 Biến môi trường

Server dùng 2 biến môi trường chính:

| Biến | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `HF_TOKEN` | ✅ | (không có — sẽ fail nhanh nếu thiếu) | HuggingFace token để tải Pyannote. Lấy tại https://huggingface.co/settings/tokens |
| `HF_HOME` | ❌ | `/workspace/cache` | Thư mục cache cho HuggingFace models (Pyannote). Trên local dev có thể đổi thành `./cache` |

> ⚠️ Server KHÔNG còn fallback hardcode cho `HF_TOKEN`. Nếu thiếu, server sẽ refuse start với log hướng dẫn cụ thể. Điều này để tránh dùng token đã thu hồi / commit nhầm lên repo.

**Cách 1 — Dùng file .env (khuyến nghị cho local dev):**

```bash
cp .env.example .env
# Sửa .env, điền HF_TOKEN thật
```

**Cách 2 — Biến môi trường shell:**

```bash
export HF_TOKEN="hf_your_token_here"   # Linux/macOS
setx HF_TOKEN "hf_your_token_here"    # Windows
```

**Cách 3 — RunPod Console (cho deployment):**

Vào RunPod Console → Serverless → Endpoint → **Environment Variables** → thêm `HF_TOKEN` và `HF_HOME`.

> ⚠️ **KHÔNG commit file `.env` lên Git** — nó đã có trong `.gitignore`. Chỉ commit `.env.example` (template không chứa secret thật).

---

## 🐛 Troubleshooting

### ❌ `Login failed: 401 Unauthorized` khi tải Pyannote
- Token `HF_TOKEN` chưa được set hoặc sai
- Chưa accept terms tại https://huggingface.co/pyannote/speaker-diarization-community-1

### ❌ `FileNotFoundError: model_vi/encoder-*.onnx`
- Chưa chạy `python builder.py` để tải models
- Hoặc `HF_HOME` trỏ sai chỗ

### ❌ `torch.cuda.OutOfMemoryError`
- File audio quá dài (>5h) — pipeline đã có VAD buffer 18000s nhưng vẫn có thể OOM
- Giảm `chunk_size` trong `ai_engine.py` (mặc định 30s) xuống 15s
- Hoặc dùng GPU lớn hơn (≥ 12 GB VRAM)

### ❌ `RuntimeError: FFmpeg not found`
- Chưa cài ffmpeg — xem [Bước 5](#5-cài-ffmpeg)

### ❌ Output `transcript: null` (FastAPI) hoặc `"transcript": []` (RunPod)
- File audio không có giọng nói, hoặc quá ngắn (< 0.5s)
- File audio chỉ có nhạc nền, không có speech

### ❌ Build Docker chậm / timeout
- Pull base image trước: `docker pull runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Tăng Docker daemon memory limit
- Build với `--progress=plain` để xem log rõ hơn

### ❌ Polling job không trả về kết quả
- FastAPI `jobs_db` là **in-memory**, mất khi restart server
- Job ID hết hạn sau khi restart
- Trên RunPod, job ID tồn tại trong vòng đời endpoint (TTL mặc định 30 ngày)

---

## 📜 License & credits

- **Sherpa-ONNX**: Apache 2.0 (k2-fsa)
- **Pyannote Audio**: MIT
- **Vietnamese Zipformer**: https://huggingface.co/zzasdf/viet_iter3_pseudo_label
- **English Zipformer (GigaSpeech)**: https://github.com/k2-fsa/sherpa-onnx
- **Silero VAD**: MIT
