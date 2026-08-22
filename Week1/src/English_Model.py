from pathlib import Path
import joblib

from Preprocessing_Pipeline import preprocess_english


class EnglishSentimentModel:

    def __init__(self):
        project_dir = Path(__file__).resolve().parent

        self.model = joblib.load(
            project_dir / "English_Model_Weights.pkl"
        )

        self.vectorizer = joblib.load(
            project_dir / "English_Model_Vectorizer.pkl"
        )

    def predict(self, text):

        cleaned_text = preprocess_english(text)

        features = self.vectorizer.transform(
            [cleaned_text]
        )

        prediction = self.model.predict(features)[0]

        return prediction