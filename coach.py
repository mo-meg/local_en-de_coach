
import json
from openai import OpenAI


def build_system_prompt(
    language,
    level,
    focus,
    scenario,
    correction_intensity,
    feedback_language,
    personal_goal,
    response_length
):

    correction_rules = {
        "Light": "Correct only important mistakes.",
        "Balanced": "Correct one or two useful mistakes.",
        "Detailed": "Explain important grammar and vocabulary mistakes."
    }

    response_rules = {
        "Short": "Use 1-2 spoken sentences.",
        "Medium": "Use 2-4 spoken sentences.",
        "Detailed": "Use 3-5 spoken sentences."
    }

    return f"""
You are a friendly, intelligent language conversation coach.

PRACTICE LANGUAGE: {language}
LEARNER LEVEL: {level}
LEARNING FOCUS: {focus}
SCENARIO: {scenario}

PERSONAL LEARNING GOAL:
{personal_goal}

FEEDBACK EXPLANATION LANGUAGE:
{feedback_language}

CORRECTION POLICY:
{correction_rules[correction_intensity]}

SPOKEN RESPONSE LENGTH:
{response_rules[response_length]}

IMPORTANT CONVERSATION RULES:

1. Speak naturally, like a real human conversation partner.
2. Always communicate in the selected practice language.
3. Respond to the meaning of the learner's message.
4. Do not turn every answer into a grammar lesson.
5. Correct mistakes gently.
6. Give a natural alternative expression when useful.
7. Ask exactly ONE follow-up question.
8. Adapt vocabulary and grammar to the learner's CEFR level.
9. Encourage the learner to speak in complete sentences.
10. Never invent pronunciation scores. ASR transcription is not pronunciation assessment.
11. Keep the conversation interesting and practical.
12. Never repeat the complete learner message unnecessarily.

RETURN ONLY VALID JSON:

{{
    "reply": "The natural spoken response and one follow-up question.",
    "correction": "Corrected learner sentence or empty string.",
    "explanation": "Brief explanation in the feedback language.",
    "vocabulary": "Up to two useful expressions and meanings."
}}

CRITICAL:
Only the reply field will be converted into speech.

Do not put headings, markdown, grammar explanations or vocabulary
inside the reply field.
"""


class CoachClient:

    def __init__(
        self,
        base_url,
        model,
        temperature=0.65
    ):

        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key="ollama",
            timeout=180
        )

        self.model = model
        self.temperature = temperature

    def reply(
        self,
        system_prompt,
        history,
        user_text
    ):

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(history[-16:])

        messages.append({
            "role": "user",
            "content": user_text
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content or ""


def parse_coach_response(raw):

    try:
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()

            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        data = json.loads(cleaned)

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")

        reply = str(data.get("reply", "")).strip()

        if not reply:
            raise ValueError("Empty coach reply")

        return {
            "reply": reply,
            "correction": str(
                data.get("correction", "")
            ).strip(),
            "explanation": str(
                data.get("explanation", "")
            ).strip(),
            "vocabulary": str(
                data.get("vocabulary", "")
            ).strip()
        }

    except Exception:

        # Fallback for smaller models that fail to return JSON.
        return {
            "reply": raw.strip() or "Could you tell me more?",
            "correction": "",
            "explanation": "",
            "vocabulary": ""
        }
