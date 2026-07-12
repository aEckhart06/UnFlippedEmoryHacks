import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def load_transcript(filepath="segments.md"):
    with open(filepath, "r") as file:
        return file.read()

def split_transcript_by_time(transcript_text, interval=300):
    segments = []
    current_block = ""
    current_start = 0

    for line in transcript_text.strip().split("\n"):
        match = re.match(r"Start: ([\d.]+)s End: ([\d.]+)s Text: (.*)", line)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3)

            if end - current_start > interval:
                segments.append(current_block)
                current_block = ""
                current_start = start

            current_block += text + " "

    if current_block:
        segments.append(current_block)

    return segments

def generate_mcq(text_segment):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                            You are a quiz-generating assistant. Create one multiple-choice question based on the provided text. Be creative. 
                            Make it a JSON object that is formatted exactly like this:
                            {{
                                "question" : "A creative question about the text", 
                                "choices" : {{
                                    "a" : "The first answer choice",
                                    "b" : "The second answer choice",
                                    "c" : "The third answer choice",
                                    "d" : "The fourth answer choice"
                                }},
                                "answer" : "x",
                                "explanation" : "A creative explanation for why the answer is correct"
                            }}
                            
                            Make sure the JSON structure is valid and formatted properly.
                            """
            },
            {
                "role": "user",
                "content": f"Text: {text_segment}"
            }
        ]
    )
    return response.choices[0].message.content

def ask_question(question_block):
    print("\n" + "-"*50)
    print(question_block.split("Answer:")[0].strip())
    
    correct_match = re.search(r"Answer:\s*([A-D])", question_block)
    explanation_match = re.search(r"Explanation:\s*(.*)", question_block)

    if not correct_match or not explanation_match:
        print("⚠️ Couldn't find answer or explanation. Skipping this question.")
        return

    correct_answer = correct_match.group(1).strip().upper()
    explanation = explanation_match.group(1).strip()

    user_answer = input("Your answer (A, B, C, or D): ").strip().upper()

    if user_answer == correct_answer:
        print("✅ Correct!")
    else:
        print(f"❌ Incorrect. The correct answer is {correct_answer}.")
        print("Explanation:", explanation)

def sanitize_json_from_llm(json_text):
    """Sanitize and repair JSON from LLM output."""
    import re
    import json
    
    # Remove any leading/trailing whitespace
    json_text = json_text.strip()
    
    # Remove code block markers if present (```json and ```)
    json_text = re.sub(r'^```json\s*', '', json_text)
    json_text = re.sub(r'\s*```$', '', json_text)
    
    # Fix trailing commas
    json_text = re.sub(r',\s*}', '}', json_text)
    json_text = re.sub(r',\s*\]', ']', json_text)
    
    # Ensure all quotes are double quotes (replace single quotes with double quotes)
    # This is tricky - we need to avoid replacing quotes inside words (like "don't")
    # A simple approach might be:
    json_text = re.sub(r'(?<=\s)\'(?=\w)', '"', json_text)  # Opening quotes
    json_text = re.sub(r'(?<=\w)\'(?=\s)', '"', json_text)  # Closing quotes
    
    # Try to parse and fix if needed
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse error: {e}")
        
        # More aggressive fixes if the first pass failed
        # Replace all single quotes with double quotes (this is risky)
        json_text = json_text.replace("'", '"')
        
        # Ensure property names are double-quoted
        json_text = re.sub(r'(\w+)\s*:', r'"\1":', json_text)
        
        # Escape any unescaped double quotes in strings
        # This is complex and might require a more sophisticated approach
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Failed to repair JSON: {e}")
            print(f"Problematic JSON: {json_text}")
            # Return a fallback object
            return {
                "question": "Sorry, I couldn't generate a valid question.",
                "choices": {
                    "a": "Technical error", 
                    "b": "Please try again",
                    "c": "Sorry for the inconvenience",
                    "d": "This is a fallback question"
                },
                "answer": "b",
                "explanation": "There was an error processing the AI-generated question."
            }

def main():
    try:
        user_input = int(input("How often should a question appear? (Enter number of minutes): ").strip())
        interval_seconds = user_input * 60
    except ValueError:
        print("Please enter a valid integer next time. Using default of 5 minutes.")
        interval_seconds = 300

    transcript = load_transcript()
    segments = split_transcript_by_time(transcript, interval=interval_seconds)

    for i, segment in enumerate(segments):
        start_minute = i * (interval_seconds // 60)
        end_minute = (i + 1) * (interval_seconds // 60)
        print(f"\n🧠 Generating question for Minute {start_minute}-{end_minute}...")
        question = generate_mcq(segment)
        ask_question(question)

    print("\n🎉 Done! Great job reviewing your video.")


if __name__ == "__main__":
    main()
