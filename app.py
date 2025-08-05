
from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("medium")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio = request.files["audio"]
    task = request.form.get("task", "transcribe")
    language = request.form.get("language", "zh")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio.save(tmp.name)
        result = model.transcribe(tmp.name, task=task, language=language)
        os.remove(tmp.name)

    return jsonify({"text": result["text"]})

@app.route("/")
def index():
    return "Whisper API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
