from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)  # 🔥 允許跨來源請求（CORS）

# 載入 Whisper 模型（你也可以改成 base, medium 等等）
model = whisper.load_model("small")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["audio"]

    # 將音檔儲存成臨時檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # Whisper 辨識
        result = model.transcribe(audio_path, language="zh")
        return jsonify({"text": result["text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)  # 清理臨時檔案

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
