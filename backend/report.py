from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

def get_report(json_report):
    prompt = f"""
    Context: You are a world-class vocal coach who must deliver
    professional feedback on a vocal performance based on JSON data.

    Analyze this file: {json_report}
    It contains NoteSeg objects, each object represents a note that the singer sang.
    They can be classified in 5 ways:
    'scale': a diatonic note (in the user's chosen key)
    'blue': a chromatic note that still sounds good musically
    'context': a chromatic note that still sounds good musically
    'passing': a quick transitional note that sounds good musically
    'out': a chromatic note that does not sound good musically

    Every NoteSeg object also contains a parameter called cents_med that
    tells you how much the artist deviated from the midi_target (in cents)

    Here is your grading rubric for cents deviation:
    -0-10: Perfect Accuracy
    -11:25: Anything in this range sounds amazing and the deviation is not
    noticeable to most listeners, even though it is technically not perfect
    -26-40: This is where the average listener starts to notice that the artist
    is going off-key
    -40-50: This will sound off-key to many listeners but can sound good in some contexts
    -51+: Anything here is definitely off-key

    I want you to generate a concise summary (5-6 sentences max), talking about
    key compliance, intonation tightness, and any noticeable trends to give feedback
    to the user. Your tone should be professional, encouraging, and technically accurate.
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    return response.text