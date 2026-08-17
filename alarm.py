import pygame
import os

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")

_alarm_playing = False
_alarm_paused = False

def play_alarm(volume=1.0):
    """Start alarm with given volume (0.0 to 1.0)."""
    global _alarm_playing, _alarm_paused
    try:
        if not _alarm_playing:
            pygame.mixer.music.load(ALARM_PATH)
            pygame.mixer.music.play(-1)
            _alarm_playing = True
            _alarm_paused = False
        elif _alarm_paused:
            pygame.mixer.music.unpause()
            _alarm_paused = False
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    except Exception as e:
        print(f"[Alarm] Error playing: {e}")

def set_alarm_volume(volume):
    """Set volume without restarting alarm (0.0 to 1.0)."""
    try:
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    except:
        pass

def pause_alarm():
    """Pause alarm playback and return True if it was paused."""
    global _alarm_paused
    try:
        if _alarm_playing and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            _alarm_paused = True
            return True
    except:
        pass
    return False

def resume_alarm(volume=None):
    """Resume alarm playback if previously paused."""
    global _alarm_paused
    try:
        if _alarm_playing and _alarm_paused:
            pygame.mixer.music.unpause()
            _alarm_paused = False
        if volume is not None:
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    except:
        pass

def stop_alarm():
    """Stop the alarm completely."""
    global _alarm_playing, _alarm_paused
    try:
        pygame.mixer.music.stop()
        _alarm_playing = False
        _alarm_paused = False
    except:
        pass

def is_alarm_playing():
    return _alarm_playing
