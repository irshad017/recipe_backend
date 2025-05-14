from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import speech_recognition as sr
from pydub import AudioSegment
import traceback
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# ---------------------------- Config --------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------- Load and Train Model -------------------
df = pd.read_csv("merged_recipes.csv")  # Ensure this file exists
features = ["protein", "fat", "calories", "sodium"]
X = df[features]
model = NearestNeighbors(n_neighbors=1)
model.fit(X)

# ---------------------------- Routes ---------------------------------

@app.route('/')
def home():
    return "✅ API is running! Recipe data loaded from merged_recipes.csv."

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file in request"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    webm_path = os.path.join(UPLOAD_FOLDER, 'audio.webm')
    wav_path = os.path.join(UPLOAD_FOLDER, 'audio.wav')

    try:
        # Save and convert audio file
        audio_file.save(webm_path)
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                # Recognize speech to text
                text = recognizer.recognize_google(audio_data)

                # Extract numeric values
                values = [int(s) for s in text.split() if s.isdigit()]

                # Ensure 4 values (protein, fat, calories, sodium)
                while len(values) < 4:
                    values.append(10)
                values = values[:4]

                # Predict nearest recipe
                distances, indices = model.kneighbors([values])
                closest_index = indices[0][0]
                matched_recipe = df.iloc[closest_index].to_dict()
                distance = float(distances[0][0])

                return jsonify({
                    "text": text,
                    "input_values": {
                        "protein": values[0],
                        "fat": values[1],
                        "calories": values[2],
                        "sodium": values[3]
                    },
                    "matched_recipe": matched_recipe,
                    "distance": distance
                })

            except sr.UnknownValueError:
                return jsonify({"error": "❌ Unable to recognize speech"}), 400
            except sr.RequestError as e:
                return jsonify({"error": f"❌ Google API error: {e}"}), 500

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "❌ Failed to process the audio file"}), 500

# ---------------------------- Run Server ------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
