import random

NEURAL_RITUALS = {
    "SAD": [
        {"task": "Gratitude Minute", "desc": "Identify 3 small things you're thankful for today.", "icon": "🙏"},
        {"task": "Gentle Movement", "desc": "Do 2 minutes of light stretching to release endorphins.", "icon": "🤸"},
        {"task": "Hydration Reset", "desc": "Drink a glass of water slowly, focusing on each sip.", "icon": "💧"}
    ],
    "HAPPY": [
        {"task": "Knowledge Sprint", "desc": "Spend 5 minutes learning one new concept or word.", "icon": "📚"},
        {"task": "Connection Call", "desc": "Send a quick appreciation text to a friend or colleague.", "icon": "📱"},
        {"task": "Creative Burst", "desc": "Doodle or write down one wild idea you have.", "icon": "🎨"}
    ],
    "ANGRY": [
        {"task": "Box Breathing", "desc": "Inhale 4s, Hold 4s, Exhale 4s, Hold 4s. Repeat 4 times.", "icon": "🌬️"},
        {"task": "Physical Release", "desc": "Do 10 jumping jacks to convert frustration into energy.", "icon": "⚡"},
        {"task": "Journal Dump", "desc": "Write down exactly what's upset you for 60 seconds, then delete it.", "icon": "📝"}
    ],
    "STRESSED": [
        {"task": "Digital Detox", "desc": "Close all tabs except the one you're working on right now.", "icon": "🚫"},
        {"task": "Focus Timer", "desc": "Set a Pomodoro for 25 minutes of deep work.", "icon": "⏳"},
        {"task": "Nature Gaze", "desc": "Look out a window for 60 seconds at the furthest object you see.", "icon": "🌿"}
    ],
    "NEUTRAL": [
        {"task": "Posture Check", "desc": "Straighten your spine and relax your shoulders.", "icon": "🧘"},
        {"task": "Future Scan", "desc": "Review your top priority for the next hour.", "icon": "🎯"},
        {"task": "Micro-Clean", "desc": "Tidy one small area of your physical workspace.", "icon": "🧹"}
    ],
    "FEAR": [
        {"task": "Grounding 5-4-3-2-1", "desc": "Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.", "icon": "🌍"},
        {"task": "Safety Mantra", "desc": "Repeat: 'I am in control of my breath and my immediate space.'", "icon": "🛡️"}
    ]
}

SMART_ACTIONS = {
    "SAD": "🌸 Soft Warm Lighting activated. Playing 'Lo-fi Chill' in the background.",
    "HAPPY": "☀️ Brightening environment. Increasing smart-bulb intensity to 90%.",
    "ANGRY": "❄️ Cooling local climate. AC set to 22°C. Playing calming binaural beats.",
    "STRESSED": "🌫️ Activating 'Forest Mist' diffuser. Muting all non-essential notifications.",
    "NEUTRAL": "✅ Environment Optimized. Energy-saving mode active."
}

def get_ritual(emotion: str) -> dict:
    emotion = emotion.upper()
    rituals = NEURAL_RITUALS.get(emotion, NEURAL_RITUALS["NEUTRAL"])
    return random.choice(rituals)

def get_smart_action(emotion: str) -> str:
    return SMART_ACTIONS.get(emotion.upper(), SMART_ACTIONS["NEUTRAL"])
