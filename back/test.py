import speech_recognition as sr
import time

# --- CONFIGURATION ---
TARGET_PHRASE = "blood blood blood"
LISTEN_DURATION_MINUTES = 200  # how long to keep mic active
PHRASE_DETECTION_WINDOW = 5  # seconds to capture each audio chunk

# --- INITIAL SETUP ---
recognizer = sr.Recognizer()

# --- START LISTENING ---
WARNING_FLAGGED = False
start_time = time.time()
end_time = start_time + LISTEN_DURATION_MINUTES * 60

print(f"\n🎙️ Listening for '{TARGET_PHRASE}' for {LISTEN_DURATION_MINUTES} minutes...\n")

with sr.Microphone() as source:  # Using default microphone
    recognizer.adjust_for_ambient_noise(source, duration=1)
    while time.time() < end_time:
        print("🎧 Listening...")
        try:
            # listen for short window
            audio = recognizer.listen(source, phrase_time_limit=PHRASE_DETECTION_WINDOW)
            text = recognizer.recognize_google(audio).lower()
            print(f"Heard: {text}")

            if TARGET_PHRASE in text:
                print("\n🚨 WARNING! Target phrase detected! 🚨\n")
                WARNING_FLAGGED = True
                break

        except sr.UnknownValueError:
            # couldn't understand audio
            continue
        except sr.RequestError as e:
            print(f"⚠️ API error: {e}")
            break

if not WARNING_FLAGGED:
    print("\n✅ Listening complete — no target phrase detected.")
