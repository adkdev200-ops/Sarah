import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
CHUNK_DURATION = 2

print("Loading model...")

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

print("Model loaded.")

audio_queue = queue.Queue()
buffer = np.empty((0,), dtype=np.float32)

stream = None


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())


def start_recording():
    global stream

    if stream is not None:
        return

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    stream.start()
    print("Listening...")


def stop_recording():
    global stream, buffer

    if stream is None:
        return

    stream.stop()
    stream.close()
    stream = None

    buffer = np.empty((0,), dtype=np.float32)

    while not audio_queue.empty():
        audio_queue.get()


start_recording()


def give_text():
    global buffer

    while True:

        chunk = audio_queue.get().flatten()
        buffer = np.concatenate([buffer, chunk])

        if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:

            segments, _ = model.transcribe(
                buffer,
                beam_size=1,
                vad_filter=True,
                language="en",
            )

            text = " ".join(seg.text for seg in segments).strip()

            buffer = np.empty((0,), dtype=np.float32)

            if text:
                return text