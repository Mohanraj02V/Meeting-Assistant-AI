from openai import OpenAI

client = OpenAI(api_key="")

def analyze_meeting(transcript):
    prompt = f"""
You are an AI Meeting Assistant.

Transcript:
{transcript}

Return:
1. Summary
2. Decisions
3. Action Items
4. Overall Sentiment (Positive / Neutral / Negative)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
