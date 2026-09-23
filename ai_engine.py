# ai_engine.py (BẢN v3 — DIARIZE + ZIPFORMER)
import os
import torch
import multiprocessing

# ==============================================================================
# [PERF] Tối ưu CPU threading
# ==============================================================================
_NUM_THREADS = min(8, multiprocessing.cpu_count())
torch.set_num_threads(_NUM_THREADS)
print(f"✅ CPU threads: {_NUM_THREADS} (cores: {multiprocessing.cpu_count()})")
# ==============================================================================

os.environ["HF_HOME"] = "/workspace/cache"

import gc
import concurrent.futures
import soundfile as sf
import numpy as np
import sherpa_onnx
import urllib.request
import tarfile
import shutil
import glob
import json
from diarize import diarize as diarize_fn
from torchaudio.functional import resample

# ==========================================
# CẤU HÌNH MODEL
# ==========================================

# ==========================================
# FIX LỖI TƯƠNG THÍCH (NUMPY & TORCHAUDIO)
# ==========================================
for alias in ("NAN", "NaN"):
    if not hasattr(np, alias):
        setattr(np, alias, np.nan)

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def _fake_list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _fake_list_audio_backends

# ==========================================
# HELPER: AUTO-DOWNLOAD ZIPFORMER MODEL
# ==========================================
def download_file(url, target_path, min_size=1024):
    print(f"⏳ Downloading {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        size = os.path.getsize(target_path)
        if size < min_size:
            print(f"❌ File too small ({size} bytes). Probable 404/Error page. Deleting...")
            os.remove(target_path)
            return False
        print(f"✅ Downloaded ({size/1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        if os.path.exists(target_path): os.remove(target_path)
        return False

def check_and_download_sherpa_models():
    # 1. Define URLs
    asr_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2"
    vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    
    # 2. Check & Download VAD (Global location)
    if not os.path.exists("model_vi/silero_vad.onnx") or os.path.getsize("model_vi/silero_vad.onnx") < 1024:
        os.makedirs("model_vi", exist_ok=True)
        if not download_file(vad_url, "model_vi/silero_vad.onnx"):
            print("⚠️ Retrying VAD with mirror...")
            download_file("https://huggingface.co/csukuangfj/silero-vad-onnx/resolve/main/silero_vad.onnx", "model_vi/silero_vad.onnx")

    # 3. Check & Download ASR (Vietnamese)
    if not glob.glob("model_vi/encoder-*.onnx"):
        filename = "asr_model_vi.tar.bz2"
        if download_file(asr_url, filename):
            print("📦 Extracting Vietnamese Zipformer ASR...")
            try:
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(".")
                
                extracted_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
                if os.path.exists(extracted_dir):
                    os.makedirs("model_vi", exist_ok=True) 
                    # Clean old onnx except VAD
                    for f in glob.glob("model_vi/*.onnx"):
                        if "silero_vad" not in f:
                            os.remove(f)
                    # Move new files
                    for f in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, f), "model_vi")
                    os.rmdir(extracted_dir)
                print("✅ Vietnamese Zipformer ASR Model Ready")
            except Exception as e:
                print(f"❌ Extraction Failed: {e}")
            finally:
                if os.path.exists(filename): os.remove(filename)


def check_and_download_en_sherpa_models():
    """Download English Zipformer model (GigaSpeech) into model_en/."""
    asr_en_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2"
    en_model_dir = "model_en"
    os.makedirs(en_model_dir, exist_ok=True)

    # 1. Copy shared VAD from model_vi if already downloaded
    vad_src = "model_vi/silero_vad.onnx"
    vad_dst = os.path.join(en_model_dir, "silero_vad.onnx")
    if os.path.exists(vad_src) and not os.path.exists(vad_dst):
        shutil.copy2(vad_src, vad_dst)
        print("✅ Copied silero_vad.onnx to model_en/")

    # 2. Check & Download English ASR
    if not glob.glob(os.path.join(en_model_dir, "encoder-*.onnx")):
        filename = "asr_model_en.tar.bz2"
        if download_file(asr_en_url, filename):
            print("📦 Extracting English Zipformer ASR...")
            try:
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(".")

                extracted_dir = "sherpa-onnx-zipformer-gigaspeech-2023-12-12"
                if os.path.exists(extracted_dir):
                    # Clean old onnx except VAD
                    for f in glob.glob(os.path.join(en_model_dir, "*.onnx")):
                        if "silero_vad" not in f:
                            os.remove(f)
                    # Move new files
                    for f in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, f), en_model_dir)
                    os.rmdir(extracted_dir)
                print("✅ English Zipformer ASR Model Ready")
            except Exception as e:
                print(f"❌ English ASR Extraction Failed: {e}")
            finally:
                if os.path.exists(filename): os.remove(filename)
        else:
            print("⚠️ English ASR model download failed. English transcription will be unavailable.")
    else:
        print("✅ English Zipformer model already present.")


# Lưu ý: KHÔNG gọi check_and_download_*() ở module load.
# - builder.py gọi trực tiếp để pre-download vào cache.
# - handler.py (qua MeetingAssistant.__init__) sẽ fail nhanh nếu model chưa có,
#   giúp phát hiện lỗi sớm thay vì download ngầm khi import.


class MeetingAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Init: Device={self.device.upper()}")

        # --- DIARIZATION (diarize library — ONNX, no HF key needed) ---
        print("✅ [INIT] Diarize library ready (ONNX, CPU-optimized)")

        # VAD config (shared between vi and en recognizers)
        vad_model_dir = "./model_vi"  # VAD onnx lives in model_vi
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = os.path.join(vad_model_dir, "silero_vad.onnx")
        vad_config.sample_rate = 16000
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.silero_vad.min_silence_duration = 0.5
        vad_config.silero_vad.window_size = 512
        # [FIX] Increase buffer to support long files (> 5 hours)
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=18000)

        # --- LOAD VIETNAMESE ZIPFORMER (ASR) ---
        print(f"⏳ [INIT] Đang load Vietnamese Zipformer (Sherpa-ONNX)...")
        try:
            model_dir = "./model_vi"
            tokens = os.path.join(model_dir, "tokens.txt")
            encoder = glob.glob(os.path.join(model_dir, "encoder-*.onnx"))[0]
            decoder = glob.glob(os.path.join(model_dir, "decoder-*.onnx"))[0]
            joiner = glob.glob(os.path.join(model_dir, "joiner-*.onnx"))[0]
            
            self.recognizer_vi = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=min(8, multiprocessing.cpu_count()),
                sample_rate=16000,
                feature_dim=80,
                decoding_method="modified_beam_search",
            )
            # Keep self.recognizer as alias for backward compat (used by process_hybrid_transcription)
            self.recognizer = self.recognizer_vi
            print("✅ [INIT] Vietnamese Zipformer OK.")
        except Exception as e:
            print(f"❌ [INIT ERROR] Vietnamese Zipformer: {e}")
            self.recognizer_vi = None
            self.recognizer = None

        # --- LOAD ENGLISH ZIPFORMER (ASR) ---
        print(f"⏳ [INIT] Đang load English Zipformer (Sherpa-ONNX)...")
        try:
            model_dir_en = "./model_en"
            tokens_en = os.path.join(model_dir_en, "tokens.txt")
            encoder_en = glob.glob(os.path.join(model_dir_en, "encoder-*.onnx"))[0]
            decoder_en = glob.glob(os.path.join(model_dir_en, "decoder-*.onnx"))[0]
            joiner_en = glob.glob(os.path.join(model_dir_en, "joiner-*.onnx"))[0]
            
            self.recognizer_en = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens_en,
                encoder=encoder_en,
                decoder=decoder_en,
                joiner=joiner_en,
                num_threads=min(8, multiprocessing.cpu_count()),
                sample_rate=16000,
                feature_dim=80,
                decoding_method="modified_beam_search",
            )
            print("✅ [INIT] English Zipformer OK.")
        except Exception as e:
            print(f"❌ [INIT ERROR] English Zipformer: {e}")
            self.recognizer_en = None

    def _flush_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_wav16k_mono(self, src_path: str, normalize: bool = True):
        # 1. Load Audio
        audio_np, sr = sf.read(str(src_path), always_2d=True)
        print(f"   SoundFile read: {audio_np.shape}, SR={sr}")
        
        # 2. Convert to Torch for Resample
        audio = torch.from_numpy(audio_np.T).float()
        
        # 3. Mono & Resample
        if audio.size(0) > 1: audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = resample(audio, sr, 16000)
            sr = 16000
            
        # 4. ROBUST NORMALIZATION
        if normalize:
            # Chuyển Tensor về lại NumPy chỉ để tính toán thống kê
            temp_np = audio.numpy() 
            abs_np = np.abs(temp_np)
            max_val = np.max(abs_np)
            
            if max_val > 0:
                # Dùng np.percentile của NumPy:
                # - Tính toán trên CPU cực khỏe.
                # - Không bị lỗi giới hạn phần tử như PyTorch.
                # - Tính chính xác trên toàn bộ dữ liệu (không cần downsample).
                quantile_val = np.percentile(abs_np, 99)
                
                if quantile_val == 0: quantile_val = max_val

                # Kích hoạt Normalization nếu âm lượng quá bé hoặc quá to (clipping)
                if quantile_val < 0.8 or quantile_val > 1.0:
                    target = 0.9
                    scale = float(target / quantile_val) 
                    
                    # Áp dụng scale vào Tensor
                    audio = audio * scale
                    audio = torch.clamp(audio, min=-1.0, max=1.0)
        
        return audio, sr

    def _post_process_diarization(self, segments):
        if not segments: return []
        
        # 1. Merge consecutive same speakers
        merged = [segments[0]]
        for s in segments[1:]:
            last = merged[-1]
            if s['speaker'] == last['speaker']:
                last['end'] = max(last['end'], s['end']) 
            else:
                merged.append(s)
        
        # 2. Fix "Sandwich" interruptions (A -> B_short -> A)
        # B_short < 1.5s
        refined = []
        i = 0
        while i < len(merged):
            curr = merged[i]
            # Check context
            if i > 0 and i < len(merged) - 1:
                prev = refined[-1] # Use already processed
                next_s = merged[i+1]
                
                duration = curr['end'] - curr['start']
                # [TUNING] Reduce threshold from 1.5s to 0.5s
                # Reason: 1.5s might swallow real words like "Vâng", "Đúng".
                # 0.5s is safe enough to remove noise blips.
                is_short = duration < 0.5
                is_sandwich = (prev['speaker'] == next_s['speaker'])
                
                if is_short and is_sandwich:
                    # Merge Middle into Prev, and then Next into Prev
                    print(f"   ✨ Smoothing: Merged short {curr['speaker']} ({duration:.1f}s) into {prev['speaker']}")
                    prev['end'] = next_s['end'] # Extend Prev to cover Context + Next
                    i += 2 # Skip curr and next (since next is now merged into prev)
                    continue
            
            refined.append(curr)
            i += 1
            
        return refined

    # 1. Pipeline: Audio -> Transcript (Zipformer + Pyannote) — PARALLEL
    def process_audio_to_transcript(self, audio_path: str, language: str = "vi"):
        print(f"🎤 [1/3] Start Operation...")

        # Select recognizer based on language
        if language == "en":
            recognizer = self.recognizer_en
            if recognizer is None:
                return {"error": "English ASR model not available."}
        else:
            recognizer = self.recognizer_vi

        if recognizer is None:
            return {"error": "Model chưa khởi tạo."}

        # Load Audio (Tensor 16k) - Robust Norm ON
        waveform_tensor, sr = self._load_wav16k_mono(audio_path, normalize=True)

        # Normalize Audio for ASR (Zipformer prefers normalized audio)
        max_val = torch.max(torch.abs(waveform_tensor))
        if max_val > 0:
            waveform_tensor = waveform_tensor / max_val * 0.9

        # Convert Tensor -> Numpy Array (1D) for Sherpa
        audio_samples = waveform_tensor.squeeze().numpy()
        total_len = len(audio_samples)

        # ======================================================================
        # [PERF] PARALLEL: Run Diarization + VAD+ASR simultaneously
        # Diarization (Pyannote) and ASR (Sherpa) are completely independent.
        # ======================================================================
        print("   ⏳ Running Diarization + ASR in PARALLEL...")

        def _run_diarization():
            print("   📢 Diarization thread started")
            # Save audio to temp file for diarize (it needs a file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_samples, 16000)
                tmp_path = tmp.name
            try:
                result = diarize_fn(tmp_path)
                diar_segs = [{"start": seg.start, "end": seg.end, "speaker": seg.speaker}
                             for seg in result.segments]
                diar_segs = self._post_process_diarization(diar_segs)
                print(f"   ✅ Diarization DONE! {result.num_speakers} speakers, {len(diar_segs)} segments")
                return diar_segs
            finally:
                os.remove(tmp_path)

        def _run_vad_asr():
            print("   📢 VAD+ASR thread started")
            # VAD
            self.vad.reset()
            speech_segments = []
            chunk_size = 30 * 16000

            for start in range(0, total_len, chunk_size):
                end = min(start + chunk_size, total_len)
                chunk = audio_samples[start:end]
                self.vad.accept_waveform(chunk)
                while not self.vad.empty():
                    seg = self.vad.pop()
                    speech_segments.append(seg)

            while not self.vad.empty():
                speech_segments.append(self.vad.pop())

            speech_segments = [s for s in speech_segments if s is not None]

            total_speech_samples = sum(len(s.samples) for s in speech_segments)
            coverage_ratio = total_speech_samples / total_len if total_len > 0 else 0
            print(f"   ℹ️ VAD: {len(speech_segments)} segments, coverage: {coverage_ratio*100:.1f}%")

            is_sparse = (total_len > 16000*30) and (coverage_ratio < 0.05)
            if len(speech_segments) == 0 or is_sparse:
                print("   ⚠️ VAD sparse → FORCE CHUNKING 30s")
                speech_segments = []
                chunk_duration = 30 * 16000
                for start_idx in range(0, total_len, chunk_duration):
                    class FakeSegment: pass
                    seg = FakeSegment()
                    seg.start = start_idx / 16000
                    end_idx = min(start_idx + chunk_duration, total_len)
                    seg.samples = audio_samples[start_idx:end_idx]
                    speech_segments.append(seg)

            # ASR
            print("   📢 ASR decoding started")
            words = []
            for i, segment in enumerate(speech_segments):
                if segment is None: continue
                pad_sec = 0.2
                seg_start_time = max(0, segment.start - pad_sec)
                current_len = len(segment.samples)
                start_idx = int(seg_start_time * 16000)
                original_duration = current_len / 16000
                seg_end_time = min(total_len/16000, segment.start + original_duration + pad_sec)
                end_idx = int(seg_end_time * 16000)

                files_samples_extended = audio_samples[start_idx:end_idx]
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, files_samples_extended)
                recognizer.decode_stream(stream)
                result = stream.result

                if not hasattr(result, 'tokens') or not hasattr(result, 'timestamps'):
                    continue

                raw_tokens = result.tokens
                raw_times = result.timestamps

                if language == "en":
                    word_buffer = ""
                    word_start = -1.0
                    word_end = -1.0
                    for j, token in enumerate(raw_tokens):
                        t_start = raw_times[j] + seg_start_time
                        next_t = raw_times[j+1] if j < len(raw_times)-1 else (raw_times[j] + 0.2)
                        t_end = next_t + seg_start_time
                        has_boundary = token.startswith('\u2581') or token.startswith(' ')
                        content = token.replace('\u2581', '').replace(' ', '').strip()
                        if not content: continue
                        if has_boundary:
                            if word_buffer:
                                words.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})
                            word_buffer = content
                            word_start = t_start
                            word_end = t_end
                        else:
                            if not word_buffer:
                                word_buffer = content; word_start = t_start; word_end = t_end
                            else:
                                word_buffer += content; word_end = t_end
                    if word_buffer:
                        words.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})
                else:
                    word_buffer = ""
                    word_start = -1.0
                    word_end = -1.0
                    vowels_str = "aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                    def is_vowel(c): return c in vowels_str
                    initials = {"b","c","d","đ","g","h","k","l","m","n","p","q","r","s","t","v","x","ch","gh","gi","kh","ng","ngh","nh","ph","qu","th","tr"}
                    for j, token in enumerate(raw_tokens):
                        t_start = raw_times[j] + seg_start_time
                        next_t = raw_times[j+1] if j < len(raw_times)-1 else (raw_times[j] + 0.2)
                        t_end = next_t + seg_start_time
                        has_boundary_marker = token.startswith(' ') or token.startswith(' ')
                        content = token.replace(' ', '').replace(' ', '').strip().lower()
                        if not content: continue
                        is_vowel_start = is_vowel(content[0]) if content else False
                        should_split = has_boundary_marker
                        if not should_split and word_buffer and is_vowel_start:
                            if word_buffer not in initials:
                                last_char = word_buffer[-1]
                                if not is_vowel(last_char):
                                    should_split = True
                                else:
                                    is_dipthong_closer = (len(content) == 1) and (content in "aeiouy")
                                    if not is_dipthong_closer:
                                        should_split = True
                        if should_split:
                            if word_buffer:
                                words.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})
                            word_buffer = content; word_start = t_start; word_end = t_end
                        else:
                            if not word_buffer:
                                word_buffer = content; word_start = t_start; word_end = t_end
                            else:
                                word_buffer += content; word_end = t_end
                    if word_buffer:
                        words.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})

                del stream
                if i % 10 == 0: self._flush_memory()

            print(f"   ✅ ASR DONE: {len(words)} words")
            return words

        # Run both in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            diar_future = executor.submit(_run_diarization)
            asr_future = executor.submit(_run_vad_asr)

            diar_segs = diar_future.result()
            final_words_stream = asr_future.result()

        self._flush_memory()

        # --- BƯỚC 3: ALIGNMENT (Ghép người nói vào từng từ) ---
        print("🎤 [3/3] Aligning Speakers & Words...")
        
        for word_obj in final_words_stream:
            w_start, w_end = word_obj['start'], word_obj['end']
            best_speaker = "SPEAKER_00"
            max_overlap = 0
            
            # Find Best Overlap
            for d_seg in diar_segs:
                intersection_start = max(w_start, d_seg['start'])
                intersection_end = min(w_end, d_seg['end'])
                overlap = max(0, intersection_end - intersection_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d_seg['speaker']
            
            # Fallback point-check
            if max_overlap == 0:
                 for d_seg in diar_segs:
                    if d_seg['start'] <= w_start <= d_seg['end']:
                        best_speaker = d_seg['speaker']
                        break
            
            word_obj['speaker'] = best_speaker

        if not final_words_stream: return []

        # --- BƯỚC 4: GỘP TỪ THÀNH CÂU (SEGMENTS) ---
        final_segments = []
        
        current_spk = final_words_stream[0]['speaker']
        current_words_buffer = [final_words_stream[0]]
        current_end = final_words_stream[0]['end']

        for i in range(1, len(final_words_stream)):
            item = final_words_stream[i]
            
            # Logic Logic ngắt câu: Khác người nói HOẶC im lặng quá 1.5 giây
            is_same_speaker = (item['speaker'] == current_spk)
            is_continuous = (item['start'] - current_end < 1.5) 

            if is_same_speaker and is_continuous:
                current_words_buffer.append(item)
                current_end = item['end']
            else:
                # Lưu câu cũ
                full_text = " ".join([w['word'] for w in current_words_buffer]).strip()
                if full_text:
                    final_segments.append({
                        "id": str(len(final_segments)),
                        "speakerId": current_spk,
                        "text": full_text,
                        "start": current_words_buffer[0]['start'],
                        "end": current_words_buffer[-1]['end'],
                        "words": current_words_buffer
                    })
                
                # Reset
                current_spk = item['speaker']
                current_words_buffer = [item]
                current_end = item['end']

        # Save last segment
        if current_words_buffer:
            full_text = " ".join([w['word'] for w in current_words_buffer]).strip()
            final_segments.append({
                "id": str(len(final_segments)),
                "speakerId": current_spk,
                "text": full_text,
                "start": current_words_buffer[0]['start'],
                "end": current_words_buffer[-1]['end'],
                "words": current_words_buffer
            })

        return final_segments

    # ==========================================
    # [NEW] HYBRID PIPELINE: External Diarization + Zipformer ASR
    # ==========================================
    def process_hybrid_transcription(self, audio_path: str, external_diarization: list):
        if external_diarization is None: external_diarization = []
        print(f"🎤 [Hybrid] Start Operation with External Diarization ({len(external_diarization)} segments)...")

        if self.recognizer is None:
            return {"error": "ASR Model chưa khởi tạo."}

        # Load Audio (Tensor 16k) - Robust Norm ON
        waveform_tensor, sr = self._load_wav16k_mono(audio_path, normalize=True)
        
        # --- BƯỚC 1: USE EXTERNAL DIARIZATION ---
        # Map fields: Bot returns { start_time, end_time, speaker } -> Internal { start, end, speaker }
        diar_segs = []
        for item in external_diarization:
            # [FIX] Handle if item is JSON string
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception as e:
                    print(f"⚠️ Failed to parse diarization item: {item} -> {e}")
                    continue

            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
            speaker = item.get("speaker", "SPEAKER_00")
            
            if start is not None and end is not None:
                diar_segs.append({
                    "start": float(start), 
                    "end": float(end), 
                    "speaker": str(speaker)
                })
        
        print(f"   ✅ Using External Diarization (Mapped {len(diar_segs)} segments).")
        
        # --- BƯỚC 2: ZIPFORMER ASR (COPY LOGIC) ---
        print(f"   ⏳ Running Zipformer ASR...")

        # Normalize Audio for ASR
        max_val = torch.max(torch.abs(waveform_tensor))
        if max_val > 0:
            waveform_tensor = waveform_tensor / max_val * 0.9
        
        audio_samples = waveform_tensor.squeeze().numpy()
        
        # Use VAD
        self.vad.reset()
        speech_segments = []
        chunk_size = 30 * 16000
        total_len = len(audio_samples)
        
        for start in range(0, total_len, chunk_size):
            end = min(start + chunk_size, total_len)
            chunk = audio_samples[start:end]
            self.vad.accept_waveform(chunk)
            while not self.vad.empty():
                speech_segments.append(self.vad.pop())
        while not self.vad.empty():
            speech_segments.append(self.vad.pop())
            
        speech_segments = [s for s in speech_segments if s is not None]

        # Calculate Speech Coverage
        total_speech_samples = sum(len(s.samples) for s in speech_segments)
        coverage_ratio = total_speech_samples / len(audio_samples) if len(audio_samples) > 0 else 0
        
        # [FALLBACK] Sparse check
        is_sparse = (len(audio_samples) > 16000*30) and (coverage_ratio < 0.05)
        if len(speech_segments) == 0 or is_sparse:
             print("⚠️ VAD Coverage too low. Switching to FORCE CHUNKING (30s)...")
             speech_segments = [] 
             sample_rate = 16000
             chunk_duration = 30 * sample_rate
             for start_idx in range(0, total_len, chunk_duration):
                 class FakeSegment: pass
                 seg = FakeSegment()
                 seg.start = start_idx / sample_rate
                 end_idx = min(start_idx + chunk_duration, total_len)
                 seg.samples = audio_samples[start_idx:end_idx]
                 speech_segments.append(seg)
        
        speech_segments = [s for s in speech_segments if s is not None]
        final_words_stream = []
        
        for i, segment in enumerate(speech_segments):
            if segment is None: continue
            
            # Padding
            pad_sec = 0.2
            sr = 16000
            seg_start_time = max(0, segment.start - pad_sec)
            
            start_idx = int(seg_start_time * sr)
            original_duration = len(segment.samples) / sr
            seg_end_time = min(total_len/sr, segment.start + original_duration + pad_sec)
            end_idx = int(seg_end_time * sr)
            
            files_samples_extended = audio_samples[start_idx:end_idx]
            
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, files_samples_extended)
            self.recognizer.decode_stream(stream)
            
            result = stream.result
            if not hasattr(result, 'tokens') or not hasattr(result, 'timestamps'): continue

            raw_tokens = result.tokens 
            raw_times = result.timestamps
            
            # Merge Logic (Same as before)
            word_buffer = ""
            word_start = -1.0
            word_end = -1.0
            
            for j, token in enumerate(raw_tokens):
                t_start = raw_times[j] + seg_start_time
                next_t = raw_times[j+1] if j < len(raw_times)-1 else (raw_times[j] + 0.2)
                t_end = next_t + seg_start_time
                
                has_boundary_marker = token.startswith(' ') or token.startswith(' ')
                content = token.replace(' ', '').replace(' ', '').strip().lower()
                if not content: continue 

                vowels_str = "aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                def is_vowel(c): return c in vowels_str
                initials = {"b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x", "ch", "gh", "gi", "kh", "ng", "ngh", "nh", "ph", "qu", "th", "tr"}
                
                is_vowel_start = is_vowel(content[0]) if content else False
                should_split = has_boundary_marker
                
                if not should_split and word_buffer and is_vowel_start:
                     if word_buffer not in initials:
                         last_char = word_buffer[-1]
                         if not is_vowel(last_char): should_split = True
                         else:
                             is_dipthong_closer = (len(content) == 1) and (content in "aeiouy")
                             if not is_dipthong_closer: should_split = True

                if should_split:
                    if word_buffer:
                        final_words_stream.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})
                    word_buffer = content
                    word_start = t_start
                    word_end = t_end
                else:
                    if not word_buffer:
                        word_buffer = content
                        word_start = t_start
                        word_end = t_end
                    else:
                        word_buffer += content
                        word_end = t_end
            
            if word_buffer:
                final_words_stream.append({"word": word_buffer, "start": word_start, "end": word_end, "speaker": "SPEAKER_00"})
            
            del stream
            if i % 10 == 0: self._flush_memory()

        self._flush_memory()

        # --- BƯỚC 3: ALIGNMENT (Hybrid) ---
        print("🎤 [Hybrid] Aligning Bot Speakers & Zipformer Words...")
        
        for word_obj in final_words_stream:
            w_start, w_end = word_obj['start'], word_obj['end']
            best_speaker = "SPEAKER_00"
            max_overlap = 0
            
            # Find Best Overlap
            for d_seg in diar_segs:
                # Ensure float types
                d_start = float(d_seg.get('start', 0))
                d_end = float(d_seg.get('end', 0))
                spk = d_seg.get('speaker', 'Unknown')
                
                intersection_start = max(w_start, d_start)
                intersection_end = min(w_end, d_end)
                overlap = max(0, intersection_end - intersection_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk
            
            if max_overlap == 0:
                 for d_seg in diar_segs:
                    d_start = float(d_seg.get('start', 0))
                    d_end = float(d_seg.get('end', 0))
                    if d_start <= w_start <= d_end:
                        best_speaker = d_seg.get('speaker', 'Unknown')
                        break
            
            word_obj['speaker'] = best_speaker

        if not final_words_stream: return []

        # --- BƯỚC 4: RE-SEGMENTATION ---
        final_segments = []
        current_spk = final_words_stream[0]['speaker']
        current_words_buffer = [final_words_stream[0]]
        current_end = final_words_stream[0]['end']

        for i in range(1, len(final_words_stream)):
            item = final_words_stream[i]
            is_same_speaker = (item['speaker'] == current_spk)
            is_continuous = (item['start'] - current_end < 1.5) 

            if is_same_speaker and is_continuous:
                current_words_buffer.append(item)
                current_end = item['end']
            else:
                full_text = " ".join([w['word'] for w in current_words_buffer]).strip()
                if full_text:
                    final_segments.append({
                        "id": str(len(final_segments)),
                        "speakerId": (current_spk if str(current_spk).startswith("SPEAKER_") else f"SPEAKER_{current_spk}"),
                        "text": full_text,
                        "start": current_words_buffer[0]['start'],
                        "end": current_words_buffer[-1]['end'],
                        "words": current_words_buffer
                    })
                current_spk = item['speaker']
                current_words_buffer = [item]
                current_end = item['end']

        if current_words_buffer:
            full_text = " ".join([w['word'] for w in current_words_buffer]).strip()
            final_segments.append({
                "id": str(len(final_segments)),
                "speakerId": (current_spk if str(current_spk).startswith("SPEAKER_") else f"SPEAKER_{current_spk}"),
                "text": full_text,
                "start": current_words_buffer[0]['start'],
                "end": current_words_buffer[-1]['end'],
                "words": current_words_buffer
            })

        return final_segments