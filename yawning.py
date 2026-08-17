import numpy as np

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection."""
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    if C == 0:
        return 0.0
    mar = (A + B) / (2.0 * C)
    return mar