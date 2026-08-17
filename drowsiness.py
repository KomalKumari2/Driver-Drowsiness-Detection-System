import numpy as np

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) for drowsiness detection."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)