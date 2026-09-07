import re
from collections import Counter
from config import (
    PAD_TOKEN,
    UNK_TOKEN,
    MAX_VOCAB_SIZE,
    MAX_LEN
)

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

def clean_text(text):
    text = str(text).lower()

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    return text.split()

# BUILD VOCABULARY
def build_vocabulary(tokenized_texts):

    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    most_common = counter.most_common(MAX_VOCAB_SIZE - 2)

    word_to_idx = { PAD_TOKEN: 0,
                    UNK_TOKEN: 1
                     }

    for word, _ in most_common:

        word_to_idx[word] = len( word_to_idx )

    return word_to_idx


# CONVERT TOKENS TO INTEGER SEQUENCE

def tokens_to_sequence(
    tokens,
    word_to_idx
):

    unk_idx = word_to_idx[UNK_TOKEN]

    return [word_to_idx.get(token,unk_idx) for token in tokens]



#PADDING
def pad_sequence( sequence,  max_len=MAX_LEN):

    if len(sequence) > max_len:

        return sequence[:max_len]

    pad_length = max_len - len(sequence)

    return sequence + [0] * pad_length