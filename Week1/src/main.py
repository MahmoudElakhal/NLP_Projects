
from pathlib import Path
import joblib

from Preprocessing_Pipeline import normalize_arabic
from Arabic_Model import ArabicSentimentModel
from English_Model import EnglishSentimentModel

class SentimentSystem:

    def __init__(self):

        project_dir = Path(__file__).resolve().parent

        # Language classifier
        self.language_model = joblib.load(
            project_dir / "Language_classifier_weights.pkl"
        )

        self.language_vectorizer = joblib.load(
            project_dir / "Language_classifier_Vectorizer.pkl"
        )

        # Sentiment models
        self.arabic_model = ArabicSentimentModel()
        self.english_model = EnglishSentimentModel()


    def detect_language(self, text):

        features = self.language_vectorizer.transform(
            [text]
        )

        language = self.language_model.predict(features)[0]

        return language


    def predict(self, text):

        language = self.detect_language(text)

        if language == "arabic":

            sentiment = self.arabic_model.predict(text)

        elif language == "English":

            sentiment = self.english_model.predict(text)

        else:

            raise ValueError(
                f"Unknown language: {language}"
            )

        return language, sentiment

if __name__ == "__main__":

    system = SentimentSystem()

    while True:

        text = input("\nEnter text: ")

        if text.lower() == "exit":
            break

        language, sentiment = system.predict(text)

        print(f"Language : {language}")
        print(f"Sentiment: {sentiment}")