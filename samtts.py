import sounddevice as sd
import queue
import sys
import json
import subprocess
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Load the model
model = Model("models/small-en-us")
rec = KaldiRecognizer(model, 16000)

# Start the stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, # blocksize changes how often the program checks for completed sentences (more cpu is used but less latency if lowered) 
                       dtype='int16',
                       channels=1, callback=callback):
    print("Listening... (Ctrl+C to stop)")
    try:
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("You said:", text)
                    subprocess.run(["./SAM/sam.exe", text])
    except KeyboardInterrupt:
        print("\nStopped.")
