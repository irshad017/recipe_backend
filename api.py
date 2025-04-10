# import pandas as pd
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split

# # Load your dataset
# df = pd.read_csv("C:\\Users\\ABDUR RAHMAN\\OneDrive\\Desktop\\Buddy\\Avinya25\\recipe\\back\\cleaned_file.csv")

# # Use available columns
# features = ["protein", "fat", "calories", "sodium"]

# # Prepare the feature matrix
# X = df[features]

# # Initialize Nearest Neighbors model
# model = NearestNeighbors(n_neighbors=1)

# # Fit the model on the full dataset
# model.fit(X)

# # --- PREDICTION PART FOR A NEW INPUT ---
# new_input = [[25, 8, 400, 600]]  # Example: protein=25, fat=8, calories=400, sodium=600
# distances, indices = model.kneighbors(new_input)
# closest_index = indices[0][0]
# matched_recipe = df.iloc[closest_index]

# print("\n📌 Closest Recipe Match:")
# print(matched_recipe)
# print("🔎 Distance to closest recipe:", distances[0][0])

# # --- ACCURACY TEST ON SPLIT DATASET ---
# # Split dataset into train and test
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # Prepare feature matrices
# X_train = train_df[features]
# X_test = test_df[features]

# # Fit model on training data
# model.fit(X_train)

# # Find distances from test points to nearest in training set
# test_distances, test_indices = model.kneighbors(X_test)

# # Calculate average distance (proxy for model quality)
# average_distance = np.mean(test_distances)
# print("\n📊 Average distance on test set (lower = better):", average_distance)

# # Calculate pseudo "accuracy" based on a distance threshold
# threshold = 50  # Customize based on your dataset range
# accurate_count = sum(d[0] < threshold for d in test_distances)
# accuracy = accurate_count / len(test_distances)

# print(f"✅ Prediction accuracy (within distance {threshold}): {accuracy:.2%}")






from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import speech_recognition as sr
from pydub import AudioSegment
import traceback
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load dataset and train model
df = pd.read_csv("C:\\Users\\ABDUR RAHMAN\\OneDrive\\Desktop\\Buddy\\Avinya25\\recipe\\back\\cleaned_file.csv")  # Make sure this CSV is in the same folder
features = ["protein", "fat", "calories", "sodium"]
X = df[features]
model = NearestNeighbors(n_neighbors=1)
model.fit(X)

@app.route('/')
def home():
    return "MODEL Suggestion & Speech to Text & Recipe Recommendation API is running!"

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
        audio_file.save(webm_path)
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)

                # Example conversion from text to features
                # Text format expected: "25 protein 8 fat 400 calories 600 sodium"
                values = [int(s) for s in text.split() if s.isdigit()]
                if len(values) != 4:
                    return jsonify({"text": text, "error": "Could not extract 4 values"}), 400

                new_input = [values]
                distances, indices = model.kneighbors(new_input)
                closest_index = indices[0][0]
                matched_recipe = df.iloc[closest_index].to_dict()
                distance = float(distances[0][0])

                return jsonify({
                    "text": text,
                    "matched_recipe": matched_recipe,
                    "distance": distance
                })
            except sr.UnknownValueError:
                return jsonify({"error": "Unable to recognize speech"}), 400
            except sr.RequestError as e:
                return jsonify({"error": f"Google API error: {e}"}), 500

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Failed to process the audio file."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
