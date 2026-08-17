"""Quick voice test — run this standalone to verify TTS works."""
import pyttsx3

try:
    engine = pyttsx3.init('sapi5')   # Windows
except:
    engine = pyttsx3.init()           # Linux / Mac

engine.setProperty('rate', 165)
engine.say("Voice system working correctly. Driver monitor is ready.")
engine.runAndWait()
print("Voice test complete.")