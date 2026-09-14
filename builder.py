# builder.py
import os
from huggingface_hub import login

from pyannote.audio import Pipeline

# ==========================================
# CẤU HÌNH (Khớp với ai_engine.py)
# ==========================================
# Đặt cache folder cố định để khi server start không phải tải lại
os.environ["HF_HOME"] = "/workspace/cache" 

# Token lấy từ biến môi trường (khuyến nghị truyền qua --build-arg).
# Nếu thiếu, build vẫn thành công nhưng sẽ skip download Pyannote —
# runtime (handler.py) sẽ tự tải lại ở job đầu tiên (chậm hơn ~3-5 phút).
HF_TOKEN = os.environ.get("HF_TOKEN")

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

def download_models():
    print("⬇️ BẮT ĐẦU TẢI MODEL VÀO CACHE (/workspace/cache)...")

    pyannote_ok = False
    zipformer_ok = False

    # 1. Đăng nhập HF
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            print("✅ Đã xác thực HuggingFace Token")
        except Exception as e:
            print(f"⚠️ Cảnh báo Login: {e}")
    else:
        print("⚠️ HF_TOKEN chưa set — sẽ skip Pyannote. Runtime sẽ tự tải khi job đầu tiên chạy.")

    # --- 2. PYANNOTE (Community-1) ---
    if HF_TOKEN:
        try:
            print(f"   [1/2] Đang tải {DIARIZATION_MODEL}...")
            # Pyannote 4.x: Cần truyền token rõ ràng để tải config
            Pipeline.from_pretrained(DIARIZATION_MODEL)
            print("   ✅ Pyannote OK")
            pyannote_ok = True
        except Exception as e:
            print(f"   ❌ Lỗi Pyannote: {e}")
    else:
        print("   [1/2] Bỏ qua Pyannote (thiếu HF_TOKEN).")

    # --- 3. ZIPFORMER (Sherpa-ONNX) ---
    try:
        print(f"   [2/2] Đang tải Zipformer (Sherpa-ONNX)...")
        # Reuse logic from ai_engine to download models
        from ai_engine import check_and_download_sherpa_models, check_and_download_en_sherpa_models
        check_and_download_sherpa_models()
        check_and_download_en_sherpa_models()
        print("   ✅ Zipformer OK")
        zipformer_ok = True
    except Exception as e:
        print(f"   ❌ Lỗi Zipformer: {e}")

    print()
    print("=" * 60)
    print(f"BUILD SUMMARY:")
    print(f"  Pyannote : {'✅' if pyannote_ok else '⚠️ SKIPPED/FAILED'}")
    print(f"  Zipformer: {'✅' if zipformer_ok else '❌ FAILED'}")
    print("=" * 60)
    if not zipformer_ok:
        # Zipformer is critical — handler.py won't work without it
        raise SystemExit(1)
    if not pyannote_ok:
        print("⚠️ Build OK nhưng Pyannote chưa sẵn. Runtime cần HF_TOKEN để tải.")
    else:
        print("🎉 BUILD THÀNH CÔNG! Cache đã sẵn sàng.")

if __name__ == "__main__":
    download_models()