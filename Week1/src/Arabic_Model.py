from pathlib import Path
import joblib

from Preprocessing_Pipeline import preprocess_arabic , normalize_arabic


class ArabicSentimentModel:

    def __init__(self):
        project_dir = Path(__file__).resolve().parent

        self.model = joblib.load(
            project_dir / "Arabic_model_Weights.pkl"
        )

        self.vectorizer = joblib.load(
                    project_dir / "Arabic_model_Vectorizer.pkl"
                )
    def predict(self, text):

        # 1. Preprocess
        cleaned_text = normalize_arabic(text)

        # 2. Convert text → TF-IDF vector
        features = self.vectorizer.transform(
            [cleaned_text]
        )

        # 3. Classify
        prediction = self.model.predict(features)[0]

        return prediction