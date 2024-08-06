import time
import random

texts = [
    "Hello, World!",
    "Testing random text output.",
    "This is a test message.",
    "Another random message.",
    "Subprocess output test.",
    "Python is fun!",
    "Keep calm and code on.",
    "Random text for testing.",
    "Output message from subprocess.",
    "Final message in 10 seconds."
]

start_time = time.time()
while time.time() - start_time < 10:
    print(random.choice(texts))
    time.sleep(random.uniform(0.5, 2.0))