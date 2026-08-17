import queue
import sys
import threading

import pygame

_speech_queue = queue.Queue()

_busy = {
    "yawn": False,
    "distract": False,
    "quiz": False,
}
_busy_lock = threading.Lock()

_preferred_backend = None
_last_failure_report = None
_backend_lock = threading.Lock()


def _is_busy(channel):
    with _busy_lock:
        return _busy[channel]


def _set_busy(channel, value):
    with _busy_lock:
        _busy[channel] = value


def _log_failure_once(message):
    global _last_failure_report
    if message != _last_failure_report:
        print(message)
        _last_failure_report = message


def _speak_with_windows_sapi(text):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        if voices.Count < 1:
            raise RuntimeError("No SAPI voices installed")
        speaker.Voice = voices.Item(0)
        speaker.Rate = 0
        speaker.Volume = 100
        speaker.Speak(text)
    finally:
        pythoncom.CoUninitialize()


def _speak_with_pyttsx3(text):
    import pyttsx3

    engine = None
    try:
        if sys.platform.startswith("win"):
            try:
                engine = pyttsx3.init("sapi5")
            except Exception:
                engine = pyttsx3.init()
        else:
            engine = pyttsx3.init()

        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def _backend_order():
    if sys.platform.startswith("win"):
        return ["windows_sapi", "pyttsx3"]
    return ["pyttsx3"]


def _speak_text(text):
    global _preferred_backend, _last_failure_report

    attempts = []
    with _backend_lock:
        order = _backend_order()
        if _preferred_backend in order:
            order = [_preferred_backend] + [name for name in order if name != _preferred_backend]

        for backend in order:
            try:
                if backend == "windows_sapi":
                    _speak_with_windows_sapi(text)
                elif backend == "pyttsx3":
                    _speak_with_pyttsx3(text)
                else:
                    raise RuntimeError(f"Unknown backend: {backend}")

                if _preferred_backend != backend:
                    print(f"[Voice] Using {backend} backend.")
                _preferred_backend = backend
                _last_failure_report = None
                return True
            except Exception as exc:
                attempts.append(f"{backend}: {exc}")
                if _preferred_backend == backend:
                    _preferred_backend = None

    _log_failure_once(
        "[Voice] All TTS backends failed. "
        + " | ".join(attempts)
        + " | Install/enable a Windows voice if speech stays unavailable."
    )
    return False


def _pause_alarm_if_needed():
    was_music = False
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            was_music = True
    except Exception:
        pass
    return was_music


def _restore_alarm_if_needed(was_music, restore_volume):
    if not was_music:
        return
    try:
        pygame.mixer.music.unpause()
        pygame.mixer.music.set_volume(max(0.0, min(1.0, restore_volume)))
    except Exception:
        pass


def _worker():
    while True:
        text, channel, restore_volume = _speech_queue.get()
        try:
            was_music = _pause_alarm_if_needed()
            _speak_text(text)
            _restore_alarm_if_needed(was_music, restore_volume)
        finally:
            _set_busy(channel, False)
            _speech_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def _enqueue(text, channel, restore_volume):
    if _is_busy(channel):
        return False
    _set_busy(channel, True)
    _speech_queue.put((text, channel, restore_volume))
    return True


def speak_yawn(text, restore_volume=1.0):
    return _enqueue(text, "yawn", restore_volume)


def speak_distract(text, restore_volume=1.0):
    return _enqueue(text, "distract", restore_volume)


def speak_quiz(text, restore_volume=1.0):
    return _enqueue(text, "quiz", restore_volume)


def speak(text, restore_volume=1.0):
    return speak_yawn(text, restore_volume)


def is_speaking_yawn():
    return _is_busy("yawn")


def is_speaking_distract():
    return _is_busy("distract")


def is_speaking_quiz():
    return _is_busy("quiz")


def is_speaking():
    with _busy_lock:
        return any(_busy.values())
