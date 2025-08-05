from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os

app = Flask(__name__)
CORS(app)

model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = "temp_audio.wav"
    file.save(filename)

    try:
        result = model.transcribe(filename, language="zh")
        text = result["text"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filename)

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
