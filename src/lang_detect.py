import re
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 42

SCRIPT_RANGES = {
    "hindi":  re.compile(r'[\u0900-\u097F]'),
    "telugu": re.compile(r'[\u0C00-\u0C7F]'),
    "tamil":  re.compile(r'[\u0B80-\u0BFF]'),
}

HINGLISH_MARKERS = {"hai", "nahi", "kya", "karo", "mera", "tumhara", "kripya", "abhi", "bhi",
                     "aur", "hoga", "kaise", "karna", "raha", "gaya", "gye", "ho", "mila",
                     "rha", "ni", "kidhr", "bs", "kat", "hua", "hu"}

TINGLISH_MARKERS = {"cheyandi", "ledu", "kuda", "naa", "meeru", "dayachesi", "chesanu",
                     "vundi", "enti", "ekkada", "inka", "padaledu", "eppudu", "avtundi",
                     "kani", "avvaledu", "nenu", "cheyalekapothunna", "cheyyali", "cheppandi",
                     "enduku", "kanipinchadam", "marchipoya", "naku", "petta", "raledu",
                     "tappuga", "ayyindi", "chupistundi", "ani", "vachindi"}

def detect_language(text: str) -> str:
    for lang, pattern in SCRIPT_RANGES.items():
        if pattern.search(text):
            return lang

    # strip punctuation before matching -- "avtundi?" needs to hit "avtundi"
    words = set(re.findall(r"[a-z']+", text.lower()))

    if words & TINGLISH_MARKERS:
        return "tinglish"
    if words & HINGLISH_MARKERS:
        return "hinglish"

    try:
        top = detect_langs(text)[0]
        return "english" if top.lang == "en" else "other"
    except Exception:
        return "english"