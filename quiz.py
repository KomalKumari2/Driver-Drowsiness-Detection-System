import random

# ----- Response database -----
RESPONSES = {
    # Tiredness / fatigue
    ("tired", "sleepy", "drowsy", "exhausted", "fatigue"):
        ["You seem quite tired. Please pull over and take a short nap — even 15 minutes helps a lot.",
         "Driving while sleepy is as dangerous as drunk driving. Find a safe spot and rest.",
         "Your body is telling you it needs rest. Please listen to it.",
         "Feeling sleepy? A cup of coffee + a 20-min nap (called a 'nap-a-ccino') works great!"],

    # Yawning
    ("yawn", "yawning"):
        ["Yawning means your brain needs more oxygen — and sleep! Time for a break.",
         "Multiple yawns = your body's SOS signal. Please rest soon.",
         "Open the window for fresh air, and plan a stop within the next 5 minutes."],

    # Distraction
    ("distracted", "phone", "unfocused", "lost focus"):
        ["Please focus on driving. Keep both eyes on the road.",
         "Put the phone down. No message is worth your life.",
         "Stay alert — your full attention belongs on the road."],

    # Break / rest
    ("break", "rest", "stop", "pull over"):
        ["Great idea! Find the nearest rest area or parking lot.",
         "Taking breaks every 2 hours on long drives is the recommended standard.",
         "A 15–20 minute rest will significantly improve your alertness."],

    # Hello / greeting
    ("hello", "hi", "hey", "good morning", "good evening"):
        ["Hello! I'm your driving assistant. Stay safe out there!",
         "Hi! How are you feeling? Let me know if you need a break reminder.",
         "Hey! Eyes on the road — I'll keep watch on your alertness."],

    # Help
    ("help", "assist", "what can you do"):
        ["I can: detect drowsiness, alert you when yawning, find nearby parking (press G), "
         "and give you quizzes to stay awake. Ask me anything!"],

    # Danger / emergency
    ("danger", "emergency", "accident", "crash"):
        ["Stay calm. If safe, pull over immediately and call emergency services.",
         "In an emergency: signal, brake gently, pull to the side, turn on hazard lights."],

    # How are you
    ("how are you", "how r u"):
        ["I'm always alert and watching over you! How about you — feeling okay?"],

    # Thanks
    ("thank", "thanks", "thank you"):
        ["You're welcome! Stay safe and reach home well.",
         "No problem! I'm here whenever you need me."],
}

# Fallback replies when no keyword matches
FALLBACK = [
    "I'm not sure about that, but remember: staying alert is the most important thing right now.",
    "Keep your focus on driving. If you need a break, just say 'rest' and I'll help.",
    "I didn't catch that. Try asking about: tiredness, breaks, parking, or safety tips.",
    "Hmm, I'm a driving safety assistant — ask me about fatigue, rest stops, or alerts!",
]


def chatbot_reply(user_text: str) -> str:
    """
    Returns the best matching response for user_text.
    Works by checking if any keyword appears in the user's message.
    """
    text = user_text.lower().strip()
    if not text:
        return "Please say something — I'm listening!"

    # Check each keyword group
    for keywords, replies in RESPONSES.items():
        for kw in keywords:
            if kw in text:
                return random.choice(replies)

    # Nothing matched
    return random.choice(FALLBACK)