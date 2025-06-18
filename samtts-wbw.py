import sounddevice as sd
import queue
import sys
import json
import subprocess
from vosk import Model, KaldiRecognizer

q = queue.Queue()
last_words = []

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

model = Model("models/small-en-us")
rec = KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=2000,  # lower = faster updates
                       dtype='int16', channels=1, callback=callback):
    print("Listening... (Ctrl+C to stop)")
    try:
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                last_words.clear()  # reset word memory
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "").strip()
                if not text:
                    continue

                current_words = text.split()
                if len(current_words) > len(last_words):
                    new_word = current_words[-1]
                    print("Word:", new_word)
                    subprocess.run(["./SAM/sam.exe", new_word])
                    last_words = current_words
    except KeyboardInterrupt:
        print("\nStopped.")
