import io
import os
import re
import wave

from openai import OpenAI

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_CHAR_LIMIT = 3500

MODES = {
    "Standard": (
        "Write a clear, well-structured summary of the material. "
        "Cover the main topics in a logical order."
    ),
    "Overview (connections)": (
        "Give a high-level overview focused on how the concepts connect. "
        "Explain the big picture and the relationships between topics rather than every detail."
    ),
    "Feynman": (
        "Explain the material in plain, simple language as if teaching a curious beginner. "
        "Use everyday analogies, avoid jargon, and make each idea intuitive."
    ),
    "Exam focus": (
        "Summarize the material as exam preparation. "
        "Stress the key points to remember and the questions most likely to be asked, with short answers."
    ),
}

DETAILS = {
    "Brief": "Keep it very concise - a few sentences capturing only the essentials.",
    "Balanced": "Use a moderate length - one or two paragraphs covering the important points.",
    "In-depth": "Be thorough - several paragraphs that explain the material in depth.",
}


def _client() -> OpenAI:
    return OpenAI()


def summarize(text: str, mode: str, detail: str) -> str:
    mode_instruction = MODES.get(mode, MODES["Standard"])
    detail_instruction = DETAILS.get(detail, DETAILS["Balanced"])

    response = _client().chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You turn study material into a spoken-style summary that will be read aloud. "
                    f"{mode_instruction} {detail_instruction} "
                    "Write flowing prose without markdown, headings, bullet points, or special characters, "
                    "so that it sounds natural when spoken."
                ),
            },
            {
                "role": "user",
                "content": f"Study material:\n\n{text}",
            },
        ],
    )

    return response.choices[0].message.content.strip()


def _split_for_tts(text: str) -> list[str]:
    if len(text) <= TTS_CHAR_LIMIT:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > TTS_CHAR_LIMIT:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current.strip():
        pieces.append(current.strip())

    return pieces


def synthesize_speech(text: str, voice: str = TTS_VOICE) -> bytes:
    client = _client()
    params = None
    frames: list[bytes] = []
    for piece in _split_for_tts(text):
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=piece,
            response_format="wav",
        )
        reader = wave.open(io.BytesIO(response.read()), "rb")
        if params is None:
            params = reader.getparams()
        frames.append(reader.readframes(reader.getnframes()))
        reader.close()

    # Re-mux into a single WAV with a correct header so browsers can play it.
    output = io.BytesIO()
    writer = wave.open(output, "wb")
    writer.setnchannels(params.nchannels)
    writer.setsampwidth(params.sampwidth)
    writer.setframerate(params.framerate)
    writer.writeframes(b"".join(frames))
    writer.close()

    return output.getvalue()
