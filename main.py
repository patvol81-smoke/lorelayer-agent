import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="LoreLayer Continuity Architect API")

# Initialize client using the cloud environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class NarrativePayload(BaseModel):
    narrative_segment: str

@app.post("/audit")
async def audit_narrative(payload: NarrativePayload):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=payload.narrative_segment)],
                ),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="HIGH",
                ),
            )
        )
        return {"status": "SUCCESS", "analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
