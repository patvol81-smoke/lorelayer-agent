from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

app = FastAPI(title="LoreLayer Multi-Agent Writer's Room")

# Define the data structure for incoming drafts
class NarrativePayload(BaseModel):
    narrative_segment: str

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=api_key)

def call_specialist_agent(client: genai.Client, text: str, persona_prompt: str) -> str:
    """Helper function to invoke a specific deep-thinking sub-agent."""
    config = types.GenerateContentConfig(
        system_instruction=persona_prompt,
        thinking_config=types.ThinkingConfig(thinking_budget=1024) # Keeps that deep-thinking active!
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro', # Leveraging the pro model for complex textual analysis
            contents=text,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Agent Error: {str(e)}"

@app.post("/audit")
async def run_writers_room(payload: NarrativePayload):
    client = get_gemini_client()
    text_to_analyze = payload.narrative_segment

    # Define the 3 distinct technical personas
    architect_prompt = (
        "You are the Continuity Architect. Your job is to aggressively audit fantasy narratives "
        "for logical consistency, magic system rules, timeline integrity, and environmental cause-and-effect. "
        "Flag anything that violates established world logic or breaks immersion."
    )

    rhythm_prompt = (
        "You are the Rhythm & Pacing Specialist. Your job is to analyze narrative velocity, tension, "
        "dialogue realism, and emotional resonance. Flag areas where the scene drags, where exposition dumps "
        "kill the momentum, or where a dramatic beats feels unearned."
    )

    wordsmith_prompt = (
        "You are the Wordsmith and Line Editor. Your job is to focus on micro-level prose. "
        "Analyze sensory engagement, descriptive freshness, active vs. passive voice, and show-don't-tell execution. "
        "Highlight weak verbs, repetitive sentence structures, or over-explained emotions."
    )

    # Execute the panel review
    print("Consulting the Continuity Architect...")
    architect_review = call_specialist_agent(client, text_to_analyze, architect_prompt)

    print("Consulting the Rhythm & Pacing Specialist...")
    rhythm_review = call_specialist_agent(client, text_to_analyze, rhythm_prompt)

    print("Consulting the Line Editor...")
    wordsmith_review = call_specialist_agent(client, text_to_analyze, wordsmith_prompt)

    # Compile the final multi-agent board report
    return {
        "status": "SUCCESS",
        "editorial_board_report": {
            "continuity_architect": architect_review,
            "rhythm_and_pacing": rhythm_review,
            "line_editor": wordsmith_review
        }
    }
