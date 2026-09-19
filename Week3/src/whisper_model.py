from faster_whisper import WhisperModel


# ============================================================
# WHISPER CONFIG
# ============================================================

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


# ============================================================
# LOAD MODEL
# ============================================================

print(
    f"Loading Whisper model "
    f"'{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})..."
)

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Whisper model loaded.")


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe(path: str) -> str:

    segments, info = model.transcribe(
        path,
        beam_size=5
    )

    print(
        f"Detected language: "
        f"{info.language} "
        f"(p={info.language_probability:.2f})"
    )

    segments = list(segments)

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()