from google import genai
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
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

    -Every NoteSeg object also contains a parameter called cents_med that
    tells you how much the artist deviated from the midi_target (in cents)

    Here is your grading rubric for cents deviation:
    -0-10: Perfect Accuracy
    -11:25: Anything in this range sounds amazing and the deviation is not
    noticeable to most listeners, even though it is technically not perfect
    -26-40: This is where the average listener starts to notice that the artist
    is going off-key
    -40-50: This will sound off-key to many listeners but can sound good in some contexts
    -51+: Anything here is definitely off-key

    -The json contains the global key compliance, which shows how many of the notes that
    the artist chose to sing(which is midi_target) actually sound musically good for the
    key they are trying to sing.
    -I've also given you the global intonation tightness, which is the median number
    of cents the artist happens to deviate from a note by.
    
    I want you to use the data I've given you to give the user a concise (5-6 sentence max)
    summary of their vocal performance. This summary should be musically relevant
    and should actually help the user improve their vocal performance next time. Use these metrics
    and numbers given to you to give the user musical insights. Remember that you are a world-class
    vocal coach that genuinely wants to help the user improve their vocals. I don't want to see you saying
    "the intonation tightness was NaN." that is not helpful for a user at all. If you see NaN, you shouldn't
    mention it and just make your summary based on the information that you have that makes sense. 
    Your tone should be encouraging, professional, constructive, and honest.

    ONLY OUPUT THE SUMMARY AND NOTHING ELSE.
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=prompt
    )
    return response.text