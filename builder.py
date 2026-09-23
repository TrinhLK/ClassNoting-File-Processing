# builder.py — v3 (no Pyannote, diarize + Zipformer only)
import os

os.environ["HF_HOME"] = "/workspace/cache"

def download_models():
    print("⬇️ BẮT ĐẦU TẢI MODEL VÀO CACHE (/workspace/cache)...")

    # --- ZIPFORMER (Sherpa-ONNX) ---
    try:
        print(f"   Đang tải Zipformer (Sherpa-ONNX)...")
        from ai_engine import check_and_download_sherpa_models, check_and_download_en_sherpa_models
        check_and_download_sherpa_models()
        check_and_download_en_sherpa_models()
        print("   ✅ Zipformer OK")
    except Exception as e:
        print(f"   ❌ Lỗi Zipformer: {e}")
        raise SystemExit(1)

    print()
    print("=" * 60)
    print("BUILD SUMMARY:")
    print("  Diarize: ✅ (ONNX, no pre-download needed)")
    print("  Zipformer: ✅")
    print("=" * 60)
    print("🎉 BUILD THÀNH CÔNG!")

if __name__ == "__main__":
    download_models()
