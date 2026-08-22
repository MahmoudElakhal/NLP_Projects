import re
import unicodedata
from farasa.segmenter import FarasaSegmenter

segmenter = FarasaSegmenter(interactive=True)

contractions = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "'s": " is",
}

def preprocess_arabic(text):

    text = normalize_arabic(text)

    text = segmenter.segment(text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_english(text):
    
    text = str(text)

    text = text.lower()

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    text = re.sub(r'\d+', ' ', text)

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text



def normalize_arabic(text):
    text = str(text)

    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    text = re.sub(r'[إأآٱ]', 'ا', text)

    text = text.replace('ى', 'ي')

    text = re.sub(r'(.)\1{2,}', r'\1', text)

    text = re.sub(
        r'[^\u0600-\u06FF\u0660-\u0669A-Za-z\s]',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text

