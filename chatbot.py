import threading
import random
import time
import difflib

# Try importing speech_recognition — graceful fallback if not installed
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[Chatbot] SpeechRecognition not installed. Voice input disabled.")
    print("[Chatbot] Install with:  pip install SpeechRecognition pyaudio")

from voice import speak_quiz, is_speaking_quiz
from alarm import pause_alarm, resume_alarm, is_alarm_playing
from quiz  import chatbot_reply


# ── Chatbot questions for drowsiness verification ─────────────
CHATBOT_QUESTIONS = [
    {
        "question": "Are you feeling sleepy right now? Say yes or no.",
        "answers": {
            "yes": [
                "yes", "yeah", "yep", "yup", "haan", "han",
                "sleepy", "tired", "i am sleepy", "feeling sleepy", "yes i am"
            ],
            "no": [
                "no", "nope", "nah", "nahi", "nahin",
                "not sleepy", "not feeling sleepy",
                "i am not sleepy", "i am not feeling sleepy", "no i am not"
            ],
        },
        "accepted": {"yes", "no"},
        "hint": "Please say YES or NO clearly."
    },
    {
        "question": "If you start feeling sleepy, will you pull over and take rest? Say yes or no.",
        "answers": {
            "yes": [
                "yes", "yeah", "yep", "yup", "haan", "han",
                "i will", "yes i will", "pull over", "take rest",
                "i will pull over", "i will take rest", "take a break", "stop the car"
            ],
            "no": [
                "no", "nope", "nah", "nahi", "nahin",
                "i will not", "no i will not", "won t", "will not"
            ],
        },
        "accepted": {"yes"},
        "hint": "Please say YES if you will pull over and take rest."
    },
]

MIC_LANGUAGES = ("en-IN", "en-US")
MIC_AMBIENT_SECONDS = 0.8
MIC_PHRASE_LIMIT = 7

# ── State ──────────────────────────────────────────────────────
_chatbot_active    = False   # True while chatbot session is running
_chatbot_passed    = False   # True if driver passed all questions
_chatbot_thread    = None
_chatbot_lock      = threading.Lock()

# Callback set by main.py — called with True/False when chatbot finishes
_on_complete_cb    = None


def is_chatbot_active():
    return _chatbot_active

def chatbot_passed():
    return _chatbot_passed


# ── Microphone listener ────────────────────────────────────────
def _normalize_text(text):
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(cleaned.split())


def _extract_transcripts(result):
    transcripts = []

    if isinstance(result, str):
        text = _normalize_text(result)
        if text:
            transcripts.append(text)
        return transcripts

    if isinstance(result, dict):
        for alt in result.get("alternative", []):
            text = _normalize_text(alt.get("transcript", ""))
            if text and text not in transcripts:
                transcripts.append(text)

    return transcripts


def _recognize_candidates(recognizer, audio):
    transcripts = []
    request_error = None

    for language in MIC_LANGUAGES:
        try:
            result = recognizer.recognize_google(audio, language=language, show_all=True)
            for text in _extract_transcripts(result):
                if text not in transcripts:
                    transcripts.append(text)

            if transcripts:
                return transcripts

            text = recognizer.recognize_google(audio, language=language)
            text = _normalize_text(text)
            if text and text not in transcripts:
                transcripts.append(text)
                return transcripts

        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            request_error = e

    if transcripts:
        return transcripts
    if request_error is not None:
        raise request_error
    raise sr.UnknownValueError()


def listen_for_answer(timeout=6):
    """
    Listen via microphone for `timeout` seconds.
    Returns recognised transcript candidates or an empty list on failure.
    """
    if not SR_AVAILABLE:
        return []

    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.energy_threshold = 120
    r.dynamic_energy_adjustment_ratio = 1.6
    r.pause_threshold = 1.0
    r.phrase_threshold = 0.2
    r.non_speaking_duration = 0.5
    r.operation_timeout = 15

    was_alarm_paused = False
    try:
        with sr.Microphone() as source:
            if is_alarm_playing():
                was_alarm_paused = pause_alarm()
            print("[Chatbot] Calibrating microphone...")
            r.adjust_for_ambient_noise(source, duration=MIC_AMBIENT_SECONDS)
            print(f"[Chatbot] Mic threshold={r.energy_threshold:.0f}")
            print("[Chatbot] Listening...")
            audio = r.listen(source, timeout=timeout, phrase_time_limit=MIC_PHRASE_LIMIT)

        transcripts = _recognize_candidates(r, audio)
        if transcripts:
            print(f"[Chatbot] Heard: {transcripts[0]}")
            if len(transcripts) > 1:
                print(f"[Chatbot] Alternatives: {', '.join(transcripts[:3])}")
        return transcripts

    except sr.WaitTimeoutError:
        print("[Chatbot] No speech detected.")
        return []
    except sr.UnknownValueError:
        print("[Chatbot] Could not understand audio.")
        return []
    except sr.RequestError as e:
        print(f"[Chatbot] Speech service error: {e}")
        return []
    except Exception as e:
        print(f"[Chatbot] Mic error: {e}")
        return []
    finally:
        if was_alarm_paused:
            resume_alarm()


def _phrase_matches(text, phrase):
    if not phrase:
        return False

    if phrase in text:
        return True

    words = text.split()
    phrase_words = phrase.split()

    if len(phrase_words) == 1:
        target = phrase_words[0]
        if target in words:
            return True
        if len(target) >= 3:
            return any(difflib.SequenceMatcher(None, word, target).ratio() >= 0.86 for word in words)
        return False

    if len(words) == len(phrase_words):
        return difflib.SequenceMatcher(None, text, phrase).ratio() >= 0.84

    return False


def _classify_answer(candidates, answer_map):
    """Return the matched answer label such as yes/no, or None."""
    normalized_candidates = [_normalize_text(text) for text in candidates if _normalize_text(text)]

    phrase_entries = []
    for label, phrases in answer_map.items():
        for phrase in phrases:
            normalized_phrase = _normalize_text(phrase)
            if normalized_phrase:
                phrase_entries.append((label, normalized_phrase))

    phrase_entries.sort(key=lambda item: len(item[1]), reverse=True)

    for text in normalized_candidates:
        for label, phrase in phrase_entries:
            if _phrase_matches(text, phrase):
                return label

    return None


# ── Main chatbot session (runs in background thread) ──────────
def _run_chatbot_session(on_complete):
    global _chatbot_active, _chatbot_passed

    print("[Chatbot] Session started.")
    _chatbot_active = True
    _chatbot_passed = False

    try:
        # Small delay so alarm starts before chatbot speaks
        time.sleep(1.0)

        # Introduction
        _tts("Drowsiness detected! Please answer these questions to stop the alarm.")
        time.sleep(0.5)

        passed = True

        for i, q_data in enumerate(CHATBOT_QUESTIONS):
            _tts(q_data["question"])
            time.sleep(0.3)

            answer = listen_for_answer(timeout=8)

            answer_label = _classify_answer(answer, q_data["answers"])
            if answer_label in q_data["accepted"]:
                print(f"[Chatbot] Matched answer: {answer_label}")
                _tts("Understood. Thank you.")
                time.sleep(0.3)
            else:
                # One retry
                _tts(f"I didn't catch that. {q_data['hint']}")
                time.sleep(0.3)
                answer = listen_for_answer(timeout=8)
                answer_label = _classify_answer(answer, q_data["answers"])

                if answer_label in q_data["accepted"]:
                    print(f"[Chatbot] Matched answer: {answer_label}")
                    _tts("Good. Understood.")
                    time.sleep(0.3)
                else:
                    _tts("Wrong answer. The alarm will continue. Please pull over safely.")
                    passed = False
                    break

        if passed:
            _tts("Great! You are alert. Alarm stopped. Drive safely and take a break soon.")
            _chatbot_passed = True
        else:
            _chatbot_passed = False

    except Exception as e:
        print(f"[Chatbot] Session error: {e}")
        _chatbot_passed = False
    finally:
        _chatbot_active = False
        print(f"[Chatbot] Session ended. Passed: {_chatbot_passed}")
        if on_complete:
            on_complete(_chatbot_passed)


def _tts(text):
    """Speak via quiz channel and wait for it to finish."""
    speak_quiz(text)
    time.sleep(0.2)
    # Wait until TTS finishes before continuing
    waited = 0
    while is_speaking_quiz() and waited < 15:
        time.sleep(0.1)
        waited += 0.1


# ── Public: start chatbot ──────────────────────────────────────
def start_chatbot(on_complete=None):
    """
    Start the voice chatbot in a background thread.
    on_complete(passed: bool) is called when done.
    Safe to call multiple times — only one session runs at a time.
    """
    global _chatbot_thread, _chatbot_active

    with _chatbot_lock:
        if _chatbot_active:
            return   # already running

        _chatbot_thread = threading.Thread(
            target=_run_chatbot_session,
            args=(on_complete,),
            daemon=True
        )
        _chatbot_thread.start()


# ── Public: get a chatbot text reply (for on-screen chat) ─────
def get_reply(user_text):
    """
    Returns a text reply using the keyword-matching engine in quiz.py.
    Used to show chatbot responses on the dashboard text box.
    """
    return chatbot_reply(user_text)
