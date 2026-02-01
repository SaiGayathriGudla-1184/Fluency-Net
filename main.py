import os
import time
import sys
import uvicorn
import asyncio
import json
import subprocess
import re
import base64
import shutil
import uuid
import requests
import jiwer
import webbrowser
import scipy.io.wavfile as wav
import urllib.parse
import io
import html
import mimetypes
import numpy as np
import ast
import sqlite3
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict
from fastapi import FastAPI, WebSocket, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import traceback
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langdetect import detect

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure stdout handles UTF-8 (crucial for Windows consoles displaying emojis)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


# Robust import for faster_whisper with helpful error message
try:
    from faster_whisper import WhisperModel, decode_audio
except ImportError:
    logger.critical("❌ Critical Error: 'faster_whisper' module not found. Python 3.10-3.12 required.")
    sys.exit(1)


from kokoro_onnx import Kokoro
import edge_tts
from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama

# Load environment variables
load_dotenv()

# Configuration Constants
SAMPLE_RATE = 16000          # Audio sample rate (Hz)
ACTIVE_TIER = "high_quality" # Options: "low_latency", "balanced", "high_quality"
SPEED = 1.0                  # Speech rate multiplier (1.0 is standard, 1.1 can be rushed)
VOICE_PROFILE = "af_heart"   # Voice character selection

# --- Tier Configuration ---
TIER_CONFIG = {
    "low_latency":  {
        "whisper_model": "base",
        "beam_size": 1 # Greedy decoding for max speed
    },
    "balanced":     {
        "whisper_model": "small",
        "beam_size": 5 # Beam search: Essential for accuracy in Indian languages (Telugu, Hindi)
    },
    "high_quality": {
        "whisper_model": "large-v3", # Upgraded to large-v3 for best Telugu/Indian language accuracy
        "beam_size": 5 # Beam search is essential for quality
    }
}

# --- Agent Architecture: Model-Based & Goal-Based ---

class AgentInternalState(BaseModel):
    """
    Model-Based Reflex Agent Component:
    Maintains an internal model of the world (user) to account for unobservable aspects
    (like historical fluency trends and emotional trajectory).
    """
    session_id: str
    history: List[dict] = [] # Conversation history
    fluency_scores: List[int] = [] # Track performance over time
    emotional_state: str = "Neutral" # Current estimated emotional state
    last_strategy: Optional[str] = None # The strategy used in the previous turn
    strategy_performance: Dict[str, float] = {} # Tracks cumulative success of strategies
    last_interaction: float = Field(default_factory=time.time) # For memory cleanup

    def update_model(self, user_input: str, assistant_output: dict):
        """Updates the internal state based on new percepts and actions."""
        
        # 1. Evaluate Previous Strategy (Learning from the past)
        current_score = 0
        if "metrics" in assistant_output and "rate" in assistant_output["metrics"]:
            current_score = assistant_output["metrics"]["rate"]
            
        if self.fluency_scores and self.last_strategy:
            prev_score = self.fluency_scores[-1]
            delta = current_score - prev_score
            
            if self.last_strategy not in self.strategy_performance:
                self.strategy_performance[self.last_strategy] = 0.0
            
            # Update performance: Positive delta means the strategy worked (fluency improved)
            # We use a simple accumulation. If a strategy consistently drops fluency, score goes negative.
            self.strategy_performance[self.last_strategy] += delta

        # Update History
        self.history.append({"user": user_input, "assistant": assistant_output.get("text", "")})
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        # Update Derived Metrics (The "Model")
        if "metrics" in assistant_output and "rate" in assistant_output["metrics"]:
            self.fluency_scores.append(assistant_output["metrics"]["rate"])
            # Limit fluency scores history to prevent unbounded growth
            if len(self.fluency_scores) > 50:
                self.fluency_scores = self.fluency_scores[-50:]
        
        if "soap" in assistant_output and "s" in assistant_output["soap"]:
            self.emotional_state = assistant_output["soap"]["s"]
            
        self.last_interaction = time.time()

class GoalBasedStrategy(BaseModel):
    """
    Goal-Based Agent Component:
    Determines the best path (strategy) toward a desired state.
    """
    current_goal: str
    tactical_instruction: str

def determine_agent_goal(state: AgentInternalState) -> GoalBasedStrategy:
    """
    Evaluates the internal model to set a specific goal for the next interaction.
    """
    # Calculate average fluency from recent history
    recent_scores = state.fluency_scores[-3:] if state.fluency_scores else []
    avg_fluency = sum(recent_scores) / len(recent_scores) if recent_scores else 0
    
    # Base Logic to determine Candidate Goal
    candidate_goal = ""
    candidate_instruction = ""

    if not state.history:
        candidate_goal = "Assessment & Rapport"
        candidate_instruction = "Establish a supportive baseline. Analyze speech patterns carefully without being overly corrective."
    elif avg_fluency < 40 or "anxious" in state.emotional_state.lower():
        candidate_goal = "Anxiety Reduction"
        candidate_instruction = "Prioritize emotional support. Use gentle, encouraging language. Suggest breathing or pausing techniques. Do NOT focus on minor errors."
    elif 40 <= avg_fluency < 80:
        candidate_goal = "Fluency Shaping"
        candidate_instruction = "Focus on specific techniques like 'Easy Onset' or 'Light Contact'. Provide constructive feedback on the specific disfluencies detected."
    else:
        candidate_goal = "Naturalness & Generalization"
        candidate_instruction = "Focus on prosody, intonation, and confidence. Challenge the user to maintain fluency with longer, more complex sentences."

    # LEARNING & ADAPTATION: Check if this strategy has failed in the past
    # If the strategy has a significantly negative performance score, switch tactics.
    if candidate_goal in state.strategy_performance:
        perf_score = state.strategy_performance[candidate_goal]
        if perf_score < -15: # Threshold: Strategy has caused cumulative regression of 15% fluency
            # Adaptive Fallback Logic
            if candidate_goal == "Fluency Shaping":
                # If shaping fails, user might be overwhelmed. Go back to support.
                candidate_goal = "Anxiety Reduction"
                candidate_instruction = "Adaptive Change: Previous shaping attempts reduced fluency. Reverting to emotional support and gentle pacing to rebuild confidence."
            elif candidate_goal == "Naturalness & Generalization":
                # If generalization fails, they aren't ready. Go back to structure.
                candidate_goal = "Fluency Shaping"
                candidate_instruction = "Adaptive Change: Generalization caused regression. Returning to structured fluency techniques."

    return GoalBasedStrategy(current_goal=candidate_goal, tactical_instruction=candidate_instruction)

# --- Persistence Layer (SQLite) ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "sessions.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT, last_interaction REAL)")
    logger.info("💽 Session database initialized.")

def load_session(session_id: str) -> Optional[AgentInternalState]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT data FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return AgentInternalState.model_validate_json(row[0])
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
    return None

def save_session(state: AgentInternalState):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions (id, data, last_interaction) VALUES (?, ?, ?)",
                     (state.session_id, state.model_dump_json(), state.last_interaction))

# --- Pydantic Models for Structured Output ---
class Metrics(BaseModel):
    words: int
    disfluencies: int
    rate: int

class SoapNotes(BaseModel):
    s: str
    o: str
    a: str
    p: str

class SpeechAnalysis(BaseModel):
    text: str = Field(..., description="The corrected, fluent version of the speech.")
    english_translation: Optional[str] = Field(None, description="English translation if input is not English.")
    metrics: Metrics
    analysis: str = Field(..., description="HTML formatted analysis.")
    suggestions: str = Field(..., description="HTML formatted suggestions.")
    soap: SoapNotes
    level: str
    classification: str
    demographics: str

# --- Embedded Knowledge Base ---
therapy_knowledge_base_content = """
dysfluency_patterns:
  - pattern: "Sound Repetition (SR)"
    description: "Repetition of phonemes or syllables (e.g., 'bu-bu-butterfly', 'li-li-like')."
    example: "p-p-paper"
  - pattern: "Word Repetition (WR)"
    description: "Repetition of whole words (e.g., 'I [I] am')."
    example: "I-I-I want to go."
  - pattern: "Prolongation (PR)"
    description: "Extended phonemes within a word (e.g., 'S[sss]ee')."
    example: "Ssssssometimes I get stuck."
  - pattern: "Block (B)"
    description: "Pauses or stoppages in speech where no sound is produced."
    example: "I want a ...... glass of water."
  - pattern: "Interjection (IN)"
    description: "Insertion of unnecessary phonemes or syllables like 'um', 'uh', or 'like'."
    example: "I, um, need to, uh, think."

language_specific_insights:
  - language: "English"
    triggers: "Plosives (p, b, t, d, k, g), fricatives (s, f), and sentence-initial vowels."
    patterns: "Whole-word repetitions and initial sound prolongations are classic indicators."
  - language: "Hindi"
    triggers: "Aspirated stops (bh, ph, th, dh), retroflex sounds (ṭ, ḍ), and conjunct consonants."
    patterns: "Part-word repetitions on initial syllables are common. Code-switching to English is frequent."
  - language: "Telugu"
    triggers: "Geminate (double) consonants (e.g., 'kk', 'pp') and agglutinated word endings."
    patterns: "Prolongation of vowels is frequent. Stuttering often breaks 'Sandhi' (word joining). Reconstruct split words carefully (e.g., 'na... na... nenu' -> 'nanu')."
  - language: "Kannada"
    triggers: "Geminate consonants, retroflex sounds."
    patterns: "Similar to Telugu, vowel endings are crucial. Watch for blocks on initial plosives."
  - language: "Tamil"
    triggers: "Retroflex sounds ('zh', 'L'), initial plosives (k, p, t)."
    patterns: "Hard blocks on initial plosives. Diglossia (formal vs colloquial) may cause hesitation."
  - language: "Malayalam"
    triggers: "Nasal clusters, retroflex sounds."
    patterns: "Stuttering may manifest as rapid repetition of the first syllable of a compound word."
  - language: "Bengali"
    triggers: "Aspirated stops (bh, ph, dh), clusters with 'r' (e.g., 'pr', 'tr')."
    patterns: "Vowel elongation is common. Repetitions often occur on the first syllable of polysyllabic words."

boli_dataset_insights:
  - common_triggers: "Stop consonants (/p/, /t/, /k/), complex clusters (/str/, /br/, /pl/), and fricatives (/f/, /sh/, /th/)."
  - observation: "Stuttering is often less frequent in spontaneous speech compared to read speech. Anxiety is higher with unfamiliar listeners."

therapeutic_techniques:
  - technique: "Pacing and Slow Rate"
    description: "Intentionally slowing down the rate of speech. This can be done by pausing between words or stretching out vowels. It gives more time for motor planning and execution."
    when_to_use: "General strategy for reducing overall stuttering frequency."
  - technique: "Easy Onset / Light Articulatory Contact"
    description: "Beginning phonation of a word gently and with less tension. For example, starting a vowel with a gentle breath ('hhhh-apple') or using light contact for consonants like 'p' or 't'."
    when_to_use: "Helpful for reducing blocks and hard glottal attacks at the beginning of words."
  - technique: "Breathing and Pausing"
    description: "Using diaphragmatic (belly) breathing and taking appropriate pauses before speaking or between phrases. This helps manage anxiety and ensures sufficient breath support."
    when_to_use: "Effective for managing physical tension and preventing blocks caused by breath-holding."
  - technique: "Continuous Phonation"
    description: "Linking words together smoothly so that the voice continues to flow with minimal interruption, reducing the number of starts and stops."
    when_to_use: "Useful for reducing instances of initial-word blocks or repetitions."

prevention_and_management:
  - strategy: "Secondary Behavior Prevention"
    description: "Identifying and reducing physical concomitants (eye blinking, head nodding) before they become ingrained."
  - strategy: "Desensitization"
    description: "Voluntary stuttering or 'bouncing' to reduce the fear of stuttering, which often fuels the severity of blocks."
  - strategy: "Relapse Prevention"
    description: "Regular practice of 'Pull-outs' (easing out of a stutter) and 'Cancellations' (pausing and re-saying the word) to maintain fluency."

"""
# --- Agent Setup (Merged from agent_client.py) ---
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
if "0.0.0.0" in ollama_host:
    print("⚠️  Warning: OLLAMA_HOST is set to 0.0.0.0. Replacing with 127.0.0.1 for client connection.")
    ollama_host = ollama_host.replace("0.0.0.0", "127.0.0.1")
logger.info(f"🔌 Connecting to Ollama at: {ollama_host}")
knowledge_agent_ai = Agent(
    model=Ollama(id="llama3.1:8b", host=ollama_host, options={"temperature": 0.0}),
    tools=[],
    instructions=dedent(f"""\
        You are an expert Speech Language Pathologist AI assistant specialized in multilingual speech therapy, capable of analyzing speech patterns in any provided language (including English, Spanish, French, German, Chinese, Hindi, and others).
        
        CORE DIRECTIVE: You must output ONLY a valid JSON object.
        CRITICAL: Do NOT hallucinate. In your analysis, ONLY cite words that appear in the user's input. Do not use words from other languages or examples.

        CHAIN OF THOUGHT REASONING:
        Before generating the JSON, you MUST provide a step-by-step reasoning process to explain your analysis.
        Enclose this reasoning strictly within <thought> and </thought> tags.
        If you wish to be conversational, include your message inside the 'analysis' field (formatted as HTML).
        Do NOT output plain text, as it breaks the speech correction pipeline.
        You utilize Chain-of-Thought reasoning to analyze speech patterns deeply before generating a response.
        Pay special attention to emotions, feelings, and implied body language to refine your demographic analysis.
        
        THERAPY KNOWLEDGE BASE:
        {therapy_knowledge_base_content}
        Use 'language_specific_insights' to tailor your analysis based on the detected language.

        IN-CONTEXT LEARNING EXAMPLES (FEW-SHOT):
        - Input (English): "I... I... I want to go."
          Output:
          <thought>
          1. Detected language: English.
          2. Identified disfluency: Word Repetition (WR) on "I".
          3. Correction strategy: Remove repetitions, keep "I want to go".
          </thought>
          Output: {{
            "text": "I want to go.",
            "english_translation": "",
            "metrics": {{"words": 5, "disfluencies": 2, "rate": 60}},
            "analysis": "<ul><li><b>Word Repetition (WR)</b> of initial pronoun 'I'.</li></ul>",
            "suggestions": "<ul><li>Practice easy onset.</li></ul>",
            "soap": {{"s": "Anxious", "o": "Repetitions", "a": "Stuttering", "p": "Therapy"}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Child"
          }}
        - Input (Hindi): "Mujhe... mujhe... paani chahiye."
          Output: {{
            "text": "Mujhe paani chahiye.",
            "english_translation": "I want water.",
            "metrics": {{"words": 4, "disfluencies": 1, "rate": 75}},
            "analysis": "<ul><li><b>Word Repetition (WR)</b> of 'mujhe'.</li></ul>",
            "suggestions": "<ul><li>Slow pacing.</li></ul>",
            "soap": {{"s": "Thirsty", "o": "Repetition", "a": "Mild Disfluency", "p": "Monitor"}},
            "level": "Intermediate",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (Telugu): "నాకు... నాకు... ఆకలిగా ఉంది."
          Output: {{
            "text": "నాకు ఆకలిగా ఉంది.",
            "english_translation": "I am hungry.",
            "metrics": {{"words": 3, "disfluencies": 1, "rate": 66}},
            "analysis": "<ul><li><b>Word Repetition (WR)</b> of 'నాకు' (me/I).</li></ul>",
            "suggestions": "<ul><li>Practice easy onset.</li></ul>",
            "soap": {{"s": "Hungry", "o": "Repetition", "a": "Mild Disfluency", "p": "Monitor"}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Child"
          }}

        Your task is to analyze the provided speech transcription and populate the following JSON schema fields accurately.

        1. "text": (String) The corrected, fluent version of the speech.
           - **GOAL**: Create a perfectly fluent, natural, and efficient sentence.
           - **REMOVE ALL**: Repetitions ("I-I-I"), prolongations ("ssss-snake"), blocks (...), interjections ("um", "uh", "like"), and false starts.
           - **REPAIR**: Fix any grammar or sentence structure that was broken by the stuttering.
           - **RECONSTRUCT**: Merge fragmented syllables into whole words.
             * Example: "u... u... umbrella" -> "umbrella"
           - **Contextual Restoration**: If the stuttering has distorted a word (e.g. "mara" instead of "hamara" due to a block), restore the intended word based on the sentence context.
             * Example: "namu... makin" -> "Namumkin" (Phonetic restoration).
             * Example: "ha ha ha ha mara ba ba barat ma ma ma han" -> "Hamara Bharat Mahan" (Correcting 'mara'->'Hamara', 'barat'->'Bharat', 'han'->'Mahan' based on the famous phrase).
           
           - If the input is mixed language (e.g., "Main market ja raha hoon"), KEEP IT MIXED ("Main market ja raha hoon").
           - The corrected text MUST be in the same language as the input.
           - CRITICAL: If the input text is in English, the output "text" MUST be in English. Do NOT translate English input to Hindi or any other language.
           - Example: "I... I... wanna go" -> "I wanna go" (NOT "I want to go").
           - For Hindi/Urdu/Indian Languages: Do NOT replace Urdu/colloquial words with pure Sanskritized/Formal versions (e.g., keep "zindagi" not "jeevan" if spoken).
           - **Dialect Preservation**: For Tamil, Telugu, Kannada, and Malayalam, do NOT convert colloquial/spoken forms (e.g., "vosthunna", "poren") to formal written forms (e.g., "vasthunnanu", "pogiren") unless the input is clearly formal. Preserve the speaker's natural register.
           - Script Consistency: Ensure the output script matches the input script (e.g. Devanagari -> Devanagari, Telugu -> Telugu, Tamil -> Tamil, Kannada -> Kannada, Malayalam -> Malayalam, Bengali -> Bengali). Do NOT transliterate to Latin/English script.
           - **Transliteration Handling**: If the input is in Latin script (e.g., "na peru..." for Telugu), the output MUST remain in Latin script. Do NOT convert it to the native script.

        2. "english_translation": (String) The English translation of the corrected "text".
           - REQUIRED if the input is not English.
           - If input is English, return an empty string.

        3. "metrics": (Dictionary) Quantitative analysis of the speech.
           - "words": (Integer) Total count of words in the original transcription (including fillers and repetitions).
           - "disfluencies": (Integer) Total count of dysfluency events detected. Explicitly count:
             * <b>SR (Sound Repetition)</b>: Part-word/syllable repetitions (e.g., "b-b-ball").
             * <b>WR (Word Repetition)</b>: Whole-word repetitions (e.g., "I-I-I").
             * <b>PR (Prolongation)</b>: Stretching sounds (e.g., "ssss-snake").
             * <b>B (Block)</b>: Silent pauses/stoppages.
             * <b>IN (Interjection)</b>: Fillers (e.g., "um", "uh").
           - "rate": (Integer) The calculated fluency score (0-100). Formula: 100 - (disfluencies / words * 100). If 0 disfluencies, rate is 100.

        4. "analysis": (String) A detailed and insightful clinical analysis of speech patterns, formatted as an HTML unordered list (<ul><li>...</li></ul>).
           - **Core Behaviors**: Identify specific dysfluency types (SR, WR, PR, B, IN) with examples.
           - **Strict Citation**: When citing examples, YOU MUST USE THE EXACT WORDS FROM THE INPUT. Do not use words from other languages.
           - **Intentional Repetition**: Distinguish between stuttering and intentional repetition for emphasis (e.g., poetic or dramatic repetition like "Tarikh par tarikh"). Do not flag these as dysfluencies.
           - **Linguistic Patterns**: Check for Boli Dataset triggers: Stop consonants (/p/, /t/, /k/), clusters (/str/, /br/), or fricatives.
           - **Contextual Insight**: Infer emotional state or cognitive load based on the content and speech rate.
           - **Formatting**: Use <b>bold tags</b> for key terms (e.g., "<b>Repetition</b>", "<b>Block</b>") to enhance readability.
           - **Clinical Feedback**: Provide professional feedback similar to a Speech-Language Pathologist, noting severity and prognosis.
           - MUST BE IN ENGLISH, regardless of the input language.

        5. "suggestions": (String) Creative and evidence-based therapeutic interventions, formatted as an HTML unordered list (<ul><li>...</li></ul>).
           - **Targeted Exercises**: Provide specific exercises addressing the identified patterns (e.g., "Light Articulatory Contact for plosive sounds").
           - **Rationale**: Briefly explain *why* the technique helps (e.g., "Reduces tension in the lips").
           - **Functional Application**: Suggest how to apply this in daily conversation (e.g., "Practice this technique while ordering coffee").
           - **Creative Drills**: Include engaging practice ideas (e.g., "Rhythmic speech practice using a metronome app").
           - **Prevention & Cure**: Include strategies for preventing secondary behaviors and managing relapse (e.g., "Desensitization to reduce fear").
           - MUST BE IN ENGLISH.

        6. "soap": (Dictionary) Clinical documentation in SOAP format.
           - "s" (Subjective): Observations about the speaker's apparent mood, anxiety level, or confidence based on the text content and disfluencies.
           - "o" (Objective): Measurable data (e.g., "Speaker exhibited X instances of repetition in a Y-word sample").
           - "a" (Assessment): Clinical interpretation (e.g., "Signs consistent with moderate developmental stuttering").
           - "p" (Plan): Recommended plan of action (e.g., "Focus on continuous phonation techniques").

        7. "level": (String) The difficulty/severity level based on disfluency rate.
           - "Beginner": High disfluency (>10%). Focus on basic breathing, slow pacing, anxiety reduction.
           - "Intermediate": Moderate disfluency (3-10%). Focus on soft onsets, continuous phonation, phrasing.
           - "Advanced": Low disfluency (<3%). Focus on intonation, prosody, public speaking confidence.
           
        IN-CONTEXT LEARNING EXAMPLES (FEW-SHOT):
        - Input (Hindi/Mixed): "ha ha ha ha mara ba ba barat ma ma ma han"
          Output: {{
            "text": "Hamara Bharat Mahan.",
            "english_translation": "Our India is Great.",
            "metrics": {{"words": 3, "disfluencies": 8, "rate": 27}},
            "analysis": "<ul><li><b>Syllable Repetition</b> on 'Ha', 'Ba', 'Ma'.</li><li><b>Word Reconstruction</b>: Merged fragmented syllables to form 'Hamara', 'Bharat', 'Mahan'.</li></ul>",
            "suggestions": "<ul><li>Practice rhythm and pacing.</li></ul>",
            "soap": {{"s": "Patriotic", "o": "Severe Repetition", "a": "Developmental Stuttering", "p": "Fluency Shaping"}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (Hindi): "H-h-hindi: म-म-मेरा न-न-नाम... अ... र-र-राहुल है, और म-म-मैं... द-द-दिल्ली से हूँ।"
          Output: {{
            "text": "Hindi: मेरा नाम राहुल है, और मैं दिल्ली से हूँ।",
            "english_translation": "My name is Rahul, and I am from Delhi.",
            "metrics": {{"words": 9, "disfluencies": 5, "rate": 60}},
            "analysis": "<ul><li><b>Sound Repetition (SR)</b> on 'म' (Ma) and 'न' (Na).</li></ul>",
            "suggestions": "<ul><li>Light articulatory contact.</li></ul>",
            "soap": {{"s": "Neutral", "o": "Repetitions", "a": "Mild Stuttering", "p": "Monitor"}},
            "level": "Intermediate",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (English): "C-c-could you... c-could you t-t-tell me w-where the... the... [block]... st-station is l-l-located?"
          Output: {{
            "text": "Could you tell me where the station is located?",
            "english_translation": "",
            "metrics": {{"words": 9, "disfluencies": 6, "rate": 50}},
            "analysis": "<ul><li><b>Block</b> before 'station'.</li><li><b>Word Repetition</b> on 'could'.</li></ul>",
            "suggestions": "<ul><li>Easy onset.</li></ul>",
            "soap": {{"s": "Anxious", "o": "Blocking", "a": "Moderate Stuttering", "p": "Therapy"}},
            "level": "Intermediate",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (Tamil): "E... en... en peyar... Raja."
          Output: {{
            "text": "En peyar Raja.",
            "english_translation": "My name is Raja.",
            "metrics": {{"words": 3, "disfluencies": 2, "rate": 60}},
            "analysis": "<ul><li><b>Word Repetition</b> on 'en'.</li><li><b>Sound Repetition</b> on 'E'.</li></ul>",
            "suggestions": "<ul><li>Pausing technique.</li></ul>",
            "soap": {{"s": "Neutral", "o": "Repetition", "a": "Mild Disfluency", "p": "Monitor"}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (Kannada): "Nanna... nanna... hesaru... Ravi."
          Output: {{
            "text": "Nanna hesaru Ravi.",
            "english_translation": "My name is Ravi.",
            "metrics": {{"words": 3, "disfluencies": 1, "rate": 66}},
            "analysis": "<ul><li><b>Word Repetition</b> on 'nanna'.</li></ul>",
            "suggestions": "<ul><li>Easy onset.</li></ul>",
            "soap": {{"s": "Calm", "o": "Repetition", "a": "Mild", "p": "Continue"}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}
        - Input (Bengali): "A... a... amar naam... Rabi."
          Output: {{
            "text": "Amar naam Rabi.",
            "english_translation": "My name is Rabi.",
            "metrics": {{"words": 3, "disfluencies": 2, "rate": 50}},
            "analysis": "<ul><li><b>Sound Repetition</b> on 'A'.</li><li><b>Word Repetition</b> on 'amar'.</li></ul>",
            "suggestions": "<ul><li>Rhythmic speech.</li></ul>",
            "soap": {{"s": "Hesitant", "o": "Repetitions", "a": "Moderate", "p": "Therapy"}},
            "level": "Intermediate",
            "classification": "Developmental Stuttering",
            "demographics": "Adult"
          }}

        8. "classification": (String) The specific type of dysfluency detected. Choose one:
           - "Developmental Stuttering": Typical childhood onset patterns (repetitions, prolongations).
           - "Neurogenic Stuttering": Associated with neurological issues (consistent disfluencies).
           - "Cluttering": Rapid, irregular rate, collapsing words.
           - "Psychogenic Stuttering": Associated with psychological trauma.
           - "Normal Non-Fluency": Occasional fillers or hesitations typical of normal speech.

        9. "demographics": (String) Inferred demographics.
           - Analyze **Accent**: Use vocabulary, syntax, and dialect to pinpoint origin.
           - Analyze **Emotions & Feelings**: Detect anxiety, confidence, frustration, or hesitation.
           - Analyze **Body Language**: Infer physical tension (e.g., from blocks/struggle behaviors) or relaxation.
           - Combine these factors to estimate age, gender, and background (e.g., "Anxious Young Adult from Midwest US", "Confident Child").

        REMEMBER: The output must be a raw JSON object starting with {{ and ending with }}.

        CRITICAL INSTRUCTIONS:
        - If the input is in a non-English language, the "text" field must remain in that SAME language. Do not translate the corrected speech to English.
        - However, "analysis", "suggestions", "soap", "level", "classification", and "demographics" MUST always be in English.

        ADAPTIVE DIFFICULTY LOGIC & THERAPEUTIC SUGGESTIONS MAPPING:

        1. LEVEL: "Beginner" (High Severity)
           - Trigger: Disfluency rate > 10%.
           - Therapeutic Focus: Foundational Control.
           - REQUIRED SUGGESTIONS (Select relevant ones):
             * Basic Breathing: Diaphragmatic breathing to establish airflow.
             * Slow Pacing: Drastically reducing speech rate to manage motor planning.
             * Anxiety Reduction: Techniques to remain calm during blocks.

        2. LEVEL: "Intermediate" (Moderate Severity)
           - Trigger: Disfluency rate between 3% and 10%.
           - Therapeutic Focus: Speech Shaping.
           - REQUIRED SUGGESTIONS (Select relevant ones):
             * Soft Onsets: Starting words gently to prevent hard glottal attacks.
             * Continuous Phonation: Keeping the voice "on" between words to prevent breaks.
             * Phrasing: Grouping words together logically.

        3. LEVEL: "Advanced" (Low Severity)
           - Trigger: Disfluency rate < 3%.
           - Therapeutic Focus: Naturalness & Confidence.
           - REQUIRED SUGGESTIONS (Select relevant ones):
             * Intonation & Prosody: Making speech sound natural and expressive rather than robotic.
             * Public Speaking Confidence: Psychological strategies for speaking in front of others.
        """),
    markdown=True,
    stream=True,
)

def extract_json_from_text(content: str, prompt_text: str) -> dict:
    """Robustly extracts and parses JSON from LLM output."""
    # 0. Remove <thought> tags (Chain of Thought) to prevent interference with JSON parsing
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()

    # 1. Strip markdown fences
    content = re.sub(r'^```[a-zA-Z0-9]*\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'```$', '', content, flags=re.MULTILINE)

    # 2. Find the JSON object (first { to last })
    start = content.find('{')
    end = content.rfind('}')
    
    if start != -1 and end != -1:
        json_str = content[start:end+1].strip()
        
        # Fix common LLM error: "rate": max(...) = 88 -> "rate": 88
        json_str = re.sub(r'"rate":\s*[^,\n}]+?=\s*(\d+)', r'"rate": \1', json_str)
        # Fix trailing commas (common LLM error) which breaks json.loads
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # NOTE: Comment stripping removed to prevent breaking URLs (e.g. http://) inside strings
        
        def clean_and_return(data):
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not a dictionary")
            # Unescape HTML content for frontend rendering
            if "analysis" in data and isinstance(data["analysis"], str):
                data["analysis"] = html.unescape(data["analysis"])
            if "suggestions" in data and isinstance(data["suggestions"], str):
                data["suggestions"] = html.unescape(data["suggestions"])
            if "text" not in data:
                data["text"] = prompt_text
            return data

        # Attempt 1: Standard JSON parse
        try:
            data = json.loads(json_str, strict=False)
            return clean_and_return(data)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Aggressive cleanup (newlines, control chars)
        # This fixes issues where newlines are literal in strings but not escaped
        json_str_clean = json_str.replace('\n', ' ').replace('\r', '')
        json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str_clean)
        try:
            data = json.loads(json_str_clean, strict=False)
            return clean_and_return(data)
        except json.JSONDecodeError:
            pass

        # Attempt 3: Python literal eval (handles single quotes, True/False/None)
        # Also handle JSON booleans (true/false/null) mixed in Python syntax
        try:
            # Replace JSON booleans with Python equivalents if they appear as keywords
            json_str_python = re.sub(r'\btrue\b', 'True', json_str)
            json_str_python = re.sub(r'\bfalse\b', 'False', json_str_python)
            json_str_python = re.sub(r'\bnull\b', 'None', json_str_python)
            
            data = ast.literal_eval(json_str_python)
            if isinstance(data, dict):
                return clean_and_return(data)
        except (ValueError, SyntaxError):
            pass

    raise ValueError("No JSON object found")

def extract_acoustic_features(audio_np: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Extracts basic acoustic features to help the AI distinguish speech patterns.
    Acts as a lightweight proxy for MFCCs to detect signal characteristics like
    struggle (energy bursts) or fricative prolongation (high ZCR).
    """
    try:
        if audio_np is None or len(audio_np) == 0:
            return {"rms": 0.0, "zcr": 0.0, "variance": 0.0}
            
        # Handle potential NaNs/Infs in the audio data
        if not np.isfinite(audio_np).all():
            audio_np = np.nan_to_num(audio_np)

        # RMS Energy (Loudness/Intensity) - Indicates vocal effort
        rms = np.sqrt(np.mean(audio_np**2))
        
        # Zero Crossing Rate (Noisiness) - High ZCR indicates fricatives (s, f, sh)
        zcr = ((audio_np[:-1] * audio_np[1:]) < 0).sum() / len(audio_np)
        
        # Signal Variance - Indicates dynamic range
        variance = np.var(audio_np)
            
        return {
            "rms": float(rms) if np.isfinite(rms) else 0.0,
            "zcr": float(zcr) if np.isfinite(zcr) else 0.0,
            "variance": float(variance) if np.isfinite(variance) else 0.0
        }
    except Exception as e:
        logger.warning(f"⚠️ Feature extraction failed: {e}")
        return {"rms": 0.0, "zcr": 0.0, "variance": 0.0}

def knowledge_agent_client(prompt: str, language: str = "English", tone: str = "Professional", session_id: str = None, acoustic_features: dict = None):
    # Sanitize prompt to prevent JSON injection
    safe_prompt = json.dumps(prompt)[1:-1]

    # --- Agent Logic: Retrieve State & Determine Goal ---
    if not session_id:
        session_id = "default"

    # Load state from DB or create new
    agent_state = load_session(session_id)
    if not agent_state:
        agent_state = AgentInternalState(session_id=session_id)

    # Cleanup old sessions from DB (Optional: run periodically)
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.execute("DELETE FROM sessions WHERE last_interaction < ?", (time.time() - 86400 * 7,))

    agent_goal = determine_agent_goal(agent_state)
    
    # Construct Prompt with Goal & History
    history_str = "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in agent_state.history[-3:]])
    
    acoustic_context = ""
    if acoustic_features:
        acoustic_context = f"\nACOUSTIC FEATURES: Energy(RMS)={acoustic_features.get('rms',0):.3f}, ZeroCrossingRate={acoustic_features.get('zcr',0):.3f}."

    full_prompt = (
        f"The user is speaking in {language}. Tone: {tone}.{acoustic_context}\n"
        f"CURRENT AGENT GOAL: {agent_goal.current_goal}\n"
        f"STRATEGY: {agent_goal.tactical_instruction}\n\n"
        f"--- HISTORY (FOR CONTEXT ONLY - DO NOT ANALYZE) ---\n{history_str}\n--- END HISTORY ---\n\n"
        f"--- CURRENT INPUT (ANALYZE THIS TEXT ONLY) ---\n{safe_prompt}\n--- END CURRENT INPUT ---\n\n"
        f"TASK: Analyze ONLY the text inside 'CURRENT INPUT' and return the result as a valid JSON object. STRICTLY PRESERVE ORIGINAL WORDS in 'text' field, only removing stutters.\nCRITICAL: In 'analysis', ONLY cite words present in the CURRENT INPUT. Do not hallucinate or analyze the history."
    )
    
    # Retry Logic: Try up to 2 times to get valid JSON
    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            response = knowledge_agent_ai.run(full_prompt, stream=False)
            # Agno with response_model returns a structured object (SpeechAnalysis)
            if hasattr(response, "content") and isinstance(response.content, SpeechAnalysis):
                return response.content.model_dump()
            
            content = response.content if hasattr(response, "content") else str(response)
            
            try:
                data = extract_json_from_text(content, prompt)
                
                # --- Agent Logic: Update Internal Model ---
                agent_state.update_model(prompt, data)
                save_session(agent_state) # Persist to DB
                
                # Update the last strategy used for the NEXT turn's evaluation
                agent_state.last_strategy = agent_goal.current_goal

                return data
            except ValueError:
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
            
            # If this was the last attempt, return the fallback
            if attempt == max_retries:
                logger.error(f"ℹ️  All retries failed. Returning fallback.")
                
                raw_content = locals().get('content', '')
                if not raw_content:
                    raw_content = str(e)
                display_content = raw_content

                # Extract analysis text if JSON-like, otherwise show raw text
                if raw_content.strip().startswith('{'):
                    match = re.search(r'"analysis":\s*"(.*?)(?<!\\)"', raw_content, re.DOTALL)
                    if match:
                        display_content = match.group(1).replace('\\"', '"').replace('\\n', '\n')

                # Preserve UI styling: If content isn't already HTML list, wrap it.
                if "<ul>" in display_content or "<li>" in display_content:
                    formatted_content = display_content
                else:
                    formatted_content = f"<ul><li>{html.escape(display_content).replace(chr(10), '</li><li>')}</li></ul>"
                return {
                    "text": prompt, 
                    "english_translation": "",
                    "metrics": {"words": len(prompt.split()), "disfluencies": 0, "rate": 0},
                    "analysis": formatted_content,
                    "suggestions": "",
                    "soap": {"s": "-", "o": "-", "a": "-", "p": "-"},
                    "level": "General",
                    "classification": "General",
                    "demographics": "General"
                }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up temporary files on startup to prevent disk clutter."""
    init_db() # Initialize DB
    if os.path.exists("temp"):
        now = time.time()
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    if os.path.getmtime(file_path) < now - 86400:
                        os.unlink(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temp file {file_path}: {e}")
    yield

app = FastAPI(title="Fluency-Net", lifespan=lifespan)
templates = Jinja2Templates(directory=".")

# Enable CORS to prevent frontend connection issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow ALL origins (safest for local dev)
    allow_credentials=False, # Disable credentials to allow wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup Static & Temp Directories ---
os.makedirs("static", exist_ok=True)
os.makedirs("temp", exist_ok=True)
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

def check_and_download_models():
    """Ensures Kokoro models are present and valid."""
    models = [
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", "kokoro-v1.0.onnx"),
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "voices-v1.0.bin")
    ]
    for url, filename in models:
        file_path = os.path.join(MODELS_DIR, filename)
        # Check if file exists and is larger than 1KB (to filter out corrupt/HTML files)
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
            logger.info(f"⬇️ Downloading missing model: {filename}...")
            try:
                temp_filename = file_path + ".tmp"
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(temp_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                shutil.move(temp_filename, file_path)
                logger.info(f"✅ Downloaded {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to download {filename}: {e}")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                if os.path.exists(file_path):
                    os.remove(file_path)

# --- Initialize Models ---
# Configure eSpeak NG for Windows if not in PATH (Required for Kokoro)
if sys.platform == "win32" and shutil.which("espeak-ng") is None:
    possible_paths = [
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"ℹ️  Found eSpeak NG at: {path}")
            os.environ["PHONEMIZER_ESPEAK_PATH"] = path
            # Also add to PATH for subprocess calls
            os.environ["PATH"] += f";{os.path.dirname(path)}"
            break

check_and_download_models()
try:
    kokoro = Kokoro(os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"), os.path.join(MODELS_DIR, "voices-v1.0.bin"))
except Exception as e:
    logger.error(f"⚠️ Kokoro initialization failed: {e}")
    logger.info("ℹ️  Deleting potentially corrupt model files. They will be re-downloaded on next run.")
    if os.path.exists(os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")): os.remove(os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"))
    if os.path.exists(os.path.join(MODELS_DIR, "voices-v1.0.bin")): os.remove(os.path.join(MODELS_DIR, "voices-v1.0.bin"))
    kokoro = None

# Initialize Whisper (Load once)
whisper_model_size = TIER_CONFIG.get(ACTIVE_TIER, {}).get("whisper_model", "base")
print(f"⏳ Loading Whisper '{whisper_model_size}' model... (This may take a few minutes on first run)")
whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

# --- Internationalization (i18n) ---
translations = {
    "en": {
        "title": "Fluency-Net",
        "header": "🎙️ Fluency-Net",
        "subheader": "Real-Time Speech-to-Speech AI Assistant",
        "start_button": "Start Recording",
        "stop_button": "Stop Recording",
        "status_ready": "Ready to connect...",
        "status_connected": "🔴 Connected & Listening...",
        "status_disconnected": "Disconnected",
        "status_stopped": "⏹️ Stopped",
        "transcription_placeholder": "Transcription will appear here...",
    },
    "bn": {
        "title": "Vocal Agent - Bengali",
        "header": "🎙️ Vocal Agent",
        "subheader": "Real-Time Speech-to-Speech AI Assistant",
        "start_button": "Start Recording",
        "stop_button": "Stop Recording",
        "status_ready": "Ready...",
        "status_connected": "Listening...",
        "status_disconnected": "Disconnected",
        "status_stopped": "Stopped",
        "transcription_placeholder": "Transcription...",
    },
    "te": {
        "title": "Vocal Agent - Telugu",
        "header": "🎙️ Vocal Agent",
        "subheader": "Real-Time Speech-to-Speech AI Assistant",
        "start_button": "Start Recording",
        "stop_button": "Stop Recording",
        "status_ready": "Ready...",
        "status_connected": "Listening...",
        "status_disconnected": "Disconnected",
        "status_stopped": "Stopped",
        "transcription_placeholder": "Transcription...",
    },
    "kn": { "title": "Vocal Agent - Kannada", "header": "🎙️ Vocal Agent", "subheader": "AI Assistant", "start_button": "Record", "stop_button": "Stop", "status_ready": "Ready", "status_connected": "Listening", "status_disconnected": "Disconnected", "status_stopped": "Stopped", "transcription_placeholder": "..." },
    "ta": { "title": "Vocal Agent - Tamil", "header": "🎙️ Vocal Agent", "subheader": "AI Assistant", "start_button": "Record", "stop_button": "Stop", "status_ready": "Ready", "status_connected": "Listening", "status_disconnected": "Disconnected", "status_stopped": "Stopped", "transcription_placeholder": "..." },
    "ml": { "title": "Vocal Agent - Malayalam", "header": "🎙️ Vocal Agent", "subheader": "AI Assistant", "start_button": "Record", "stop_button": "Stop", "status_ready": "Ready", "status_connected": "Listening", "status_disconnected": "Disconnected", "status_stopped": "Stopped", "transcription_placeholder": "..." },
    "hi": {
        "title": "वोकल एजेंट - स्टटर2फ्लुएंट",
        "header": "🎙️ वोकल एजेंट",
        "subheader": "रियल-टाइम स्पीच-टू-स्पीच एआई असिस्टेंट",
        "start_button": "रिकॉर्डिंग शुरू करें",
        "stop_button": "रिकॉर्डिंग रोकें",
        "status_ready": "कनेक्ट करने के लिए तैयार...",
        "status_connected": "🔴 कनेक्टेड और सुन रहा है...",
        "status_disconnected": "डिस्कनेक्ट हो गया",
        "status_stopped": "⏹️ रुक गया",
        "transcription_placeholder": "ट्रांसक्रिप्शन यहाँ दिखाई देगा...",
    }
}

@app.get("/")
async def get(request: Request, lang: str = "en"):
    lang_code = lang if lang in translations else "en"
    return templates.TemplateResponse("index.html", {"request": request, "translations": translations[lang_code]})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/api/process")
async def api_process(
    file: UploadFile = File(...),
    voice: str = Form(None),
    tone: str = Form("Professional"),
    speed: float = Form(1.0),
    language: str = Form("auto"),
    session_id: str = Form(None)
):
    # Check if this is a regeneration request.
    # We check content_type OR if the filename ends with .txt (fallback)
    is_text_input = file.content_type == 'text/plain' or file.filename.endswith('.txt')
    
    if is_text_input:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded text file is empty.")
        return await process_audio_pipeline(file_bytes, language, voice, speed, tone, session_id, is_text_input=True)
    
    # For audio, save to temp file to handle large sizes without memory limits
    # Ensure filename has extension (crucial for blobs/recordings)
    filename = os.path.basename(file.filename)
    if not os.path.splitext(filename)[1]:
        ext = mimetypes.guess_extension(file.content_type)
        if ext:
            filename += ext
    temp_filename = f"temp/{uuid.uuid4()}_{filename}"
    try:
        loop = asyncio.get_running_loop()
        def save_file_blocking():
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        await loop.run_in_executor(None, save_file_blocking)
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

async def tts_router(text: str, lang: str, voice_id: str = None, speed: float = SPEED) -> Optional[str]:
    """Selects a TTS engine based on language and returns a URL to the generated audio file."""
    logger.info(f"🎤 Routing TTS for language: {lang}, Voice: {voice_id}")
    lang = lang.strip().lower()
    
    if not text or not text.strip():
        print("⚠️ TTS received empty text. Skipping generation.")
        return None

    # Validation: Ensure voice matches language. If non-English, discard English-specific voices.
    if voice_id and lang != "en":
        if voice_id.startswith("af_") or (voice_id.startswith("en-") and not lang.startswith("en")):
             voice_id = None

    # Determine Voice
    if not voice_id:
        # Default voice mapping
        defaults = {
            "en": "af_heart",
            "hi": "hi-IN-MadhurNeural",
            "bn": "bn-IN-BashkarNeural",
            "te": "te-IN-MohanNeural",
            "ta": "ta-IN-ValluvarNeural",
            "kn": "kn-IN-GaganNeural",
            "ml": "ml-IN-MidhunNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-KillianNeural",
            "zh": "zh-CN-YunxiNeural",
            "ja": "ja-JP-KeitaNeural",
            "ru": "ru-RU-DmitryNeural",
        }
        voice_id = defaults.get(lang, "af_heart")

    # Check if we should use Kokoro (English only, model loaded, and voice is a Kokoro voice)
    use_kokoro = lang == "en" and kokoro is not None and voice_id.startswith("af_")

    audio_data = b""
    mime_type = "audio/wav"

    if use_kokoro:
        logger.info(f"   Using Kokoro TTS (English) - {voice_id}")
        # Run blocking Kokoro call in executor
        loop = asyncio.get_running_loop()
        samples, sample_rate = await loop.run_in_executor(None, lambda: kokoro.create(text, voice=voice_id, speed=speed, lang="en-us"))
        
        # Write to in-memory buffer
        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, samples)
        audio_data = buffer.getvalue()
    else:
        # Fallback: If voice_id is a Kokoro voice but we can't use Kokoro, switch to Edge-TTS default
        if voice_id.startswith("af_"):
            voice_id = "en-US-ChristopherNeural"
            
        # Use edge-tts for other languages
        logger.info(f"   Using edge-tts: {voice_id}")
        rate_str = f"{int((speed - 1.0) * 100):+d}%"
        try:
            communicate = edge_tts.Communicate(text, voice_id, rate=rate_str)
            # Stream to memory
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            mime_type = "audio/mpeg" # EdgeTTS outputs mp3 by default
        except edge_tts.exceptions.NoAudioReceived:
            logger.warning(f"⚠️ EdgeTTS NoAudioReceived with voice {voice_id}. Trying fallback voice...")
            
            # Smart Fallback: Try swapping gender/voice if the primary fails
            fallbacks = {
                "te-IN-MohanNeural": "te-IN-ShrutiNeural",
                "ta-IN-ValluvarNeural": "ta-IN-PallaviNeural",
                "kn-IN-GaganNeural": "kn-IN-SapnaNeural",
                "ml-IN-MidhunNeural": "ml-IN-SobhanaNeural",
                "hi-IN-MadhurNeural": "hi-IN-SwaraNeural"
            }
            
            fallback_voice = fallbacks.get(voice_id)
            if fallback_voice:
                # STRICT GENDER CHECK: Only allow fallback if gender matches
                if get_voice_gender(fallback_voice) != get_voice_gender(voice_id):
                    logger.warning(f"⚠️ Fallback voice {fallback_voice} has different gender. Skipping to prevent confusion.")
                    return None

                logger.info(f"🔄 Retrying with fallback voice: {fallback_voice}")
                try:
                    communicate = edge_tts.Communicate(text, fallback_voice, rate=rate_str)
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    mime_type = "audio/mpeg"
                except Exception as e2:
                    logger.error(f"❌ Fallback EdgeTTS failed: {e2}")
                    return None
            else:
                return None
        except Exception as e:
            logger.error(f"❌ EdgeTTS failed: {e}")
            return None
        
    if audio_data:
        ext = ".mp3" if mime_type == "audio/mpeg" else ".wav"
        filename = f"{uuid.uuid4()}{ext}"
        filepath = f"temp/{filename}"
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return f"/temp/{filename}"
    return None

@app.post("/process/upload")
async def process_upload(
    file: UploadFile = File(...), 
    language: str = Form("auto"),
    voice: str = Form(None),
    speed: float = Form(None),
    tone: str = Form("Professional"),
    session_id: str = Form(None)
):
    # Ensure filename has extension
    filename = os.path.basename(file.filename)
    if not os.path.splitext(filename)[1]:
        ext = mimetypes.guess_extension(file.content_type)
        if ext:
            filename += ext
            
    # Check if text file
    is_text = filename.lower().endswith(".txt") or file.content_type == "text/plain"
    
    temp_filename = f"temp/{uuid.uuid4()}_{filename}"
    try:
        loop = asyncio.get_running_loop()
        def save_file_blocking():
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
            logger.info(f"📂 Received upload: {filename} ({size_mb:.2f} MB)")
        await loop.run_in_executor(None, save_file_blocking)
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id, is_text_input=is_text)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/process/url")
async def process_url(
    url: str = Form(...), 
    language: str = Form("auto"),
    voice: str = Form(None),
    speed: float = Form(None),
    tone: str = Form("Professional"),
    session_id: str = Form(None)
):
    # Check for YouTube URLs which return HTML, not media
    if "youtube.com" in url or "youtu.be" in url:
        raise HTTPException(status_code=400, detail="YouTube URLs are not directly supported. Please provide a direct link to an audio/video file (e.g., ending in .mp3, .mp4).")

    # Try to preserve extension from URL for correct MIME type detection
    parsed_url = urllib.parse.urlparse(url)
    path_filename = os.path.basename(parsed_url.path)
    name, ext = os.path.splitext(path_filename)
    if not ext:
        ext = ".mp4" # Default to mp4 if unknown
    temp_filename = f"temp/{uuid.uuid4()}_{name}{ext}"
    try:
        loop = asyncio.get_running_loop()
        def download_stream():
            # Use a browser-like User-Agent to avoid 403 Forbidden on some direct links
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            # Add size limit check (e.g., 50MB)
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                
                content_length = r.headers.get('content-length')
                if content_length and int(content_length) > 50 * 1024 * 1024:
                    raise ValueError("File too large (Max 50MB)")
                
                # Validate Content-Type to prevent downloading HTML pages as media
                content_type = r.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    raise ValueError("The provided URL points to a webpage, not a media file. Please provide a direct link.")
                
                with open(temp_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        await loop.run_in_executor(None, download_stream)
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id)
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/process/regenerate")
async def regenerate(
    text: str = Form(...),
    lang: str = Form(...),
    voice: str = Form(None)
):
    audio_url = await tts_router(text, lang, voice)
    return {"audio_url": audio_url}

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    path = f"temp/{filename}"
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    return {"status": "not found"}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "bn": "Bengali",
    "kn": "Kannada",
    "ta": "Tamil",
    "ml": "Malayalam",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian"
}

INITIAL_PROMPTS = {
    "hi": "नमस्ते, Namaste, म... म... मेरा नाम... Mera naam Rahul hai.",
    "te": "నమస్కారం. నా... నా... పేరు రవి. Namaskaram, Na... na... peru Ravi.",
    "ta": "வணக்கம். என்... என்... பெயர் ராஜா. Vanakkam, En... en... peyar Raja.",
    "kn": "ನಮಸ್ಕಾರ. ನನ್ನ... ನನ್ನ... ಹೆಸರು ರವಿ. Namaskara, Nanna... nanna... hesaru Ravi.",
    "ml": "നമസ്കാരം. എന്റെ... എന്റെ... പേര് രാഹുൽ. Namaskaram, Ente... ente... peru Rahul.",
    "bn": "নমস্কার, Namaskar, আ... আ... আমার নাম... Amar naam Rabi.",
    "en": "Umm, I... I... Hello, my name is... my name is John.",
    "es": "Hola, me... me... me llamo... me llamo Juan.",
    "fr": "Bonjour, je... je... je m'appelle... je m'appelle Pierre.",
    "de": "Hallo, ich... ich... ich heiße... ich heiße Hans.",
    "zh": "你好，我... 我... 我的名字是... 李明。",
    "ja": "こんにちは、私... 私... 私の名前は... 田中... 田中です。",
    "ru": "Привет, меня... меня... меня зовут... Иван."
}

SUPPORTED_EXTENSIONS = {
    # Audio
    '.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg', '.opus', '.wma',
    # Video
    '.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v', '.3gp'
}

async def process_audio_pipeline(input_data: Union[bytes, str], lang_pref, voice_pref, speed_pref=None, tone_pref="Professional", session_id=None, is_text_input=False):
    # Initialize URLs early for error safety across the entire function scope
    input_audio_url = None
    acoustic_features = {"rms": 0.0, "zcr": 0.0, "variance": 0.0}

    try:
        loop = asyncio.get_running_loop()
        
        if isinstance(input_data, str) and os.path.exists(input_data):
            # Check for supported format if not text input
            if not is_text_input:
                _, ext = os.path.splitext(input_data)
                if ext.lower() not in SUPPORTED_EXTENSIONS and ext.lower() != ".txt":
                    raise ValueError(f"Unsupported file format: {ext}. Please upload a valid audio or video file.")

            # Convert system path to web URL (e.g. temp/file.mp4 -> /temp/file.mp4)
            web_path = "/" + input_data.replace("\\", "/")
            
            input_audio_url = web_path

        if is_text_input:
            # 1. Bypass Transcription for Text Input (Regeneration)
            if isinstance(input_data, bytes):
                transcribed_text = input_data.decode('utf-8')
            else:
                with open(input_data, "r", encoding="utf-8") as f:
                    transcribed_text = f.read()
            detected_lang = lang_pref if lang_pref != "auto" else "en" # Default or trust pref
        else:
            # 1. Transcribe Audio
            
            # Decode audio to numpy array upfront to enable Acoustic Feature Extraction for files
            audio_np = None
            try:
                audio_source_for_decode = input_data
                if isinstance(input_data, bytes):
                    audio_source_for_decode = io.BytesIO(input_data)
                
                # decode_audio handles file paths (str) and file-like objects (BytesIO)
                audio_np = decode_audio(audio_source_for_decode, sampling_rate=16000)
                
                # Extract Acoustic Features (RMS, ZCR)
                acoustic_features = extract_acoustic_features(audio_np)
                print(f"📊 Acoustic Features Extracted: RMS={acoustic_features.get('rms',0):.3f}, ZCR={acoustic_features.get('zcr',0):.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Acoustic Feature Extraction failed (Audio decoding error): {e}")

            tiers_to_try = [ACTIVE_TIER]
            if ACTIVE_TIER == "low_latency":
                tiers_to_try.append("balanced")
            elif ACTIVE_TIER == "balanced":
                tiers_to_try.append("high_quality") # Fallback: Try beam search if greedy fails
            
            transcribed_text = ""
            detected_lang = "en"
            segments = []
            info = None

            for attempt_tier in tiers_to_try:
                def transcribe_blocking(tier):
                    tier_settings = TIER_CONFIG.get(tier, TIER_CONFIG["low_latency"])
                    current_beam_size = tier_settings.get("beam_size", 1)
                    
                    # --- ISOLATION: File Processing Specific Logic ---
                    # For non-English languages, we enforce a higher beam size to ensure accuracy on stuttered speech.
                    # This overrides the tier setting specifically for this pipeline.
                    if lang_pref and lang_pref not in ["auto", "en"]:
                        current_beam_size = max(current_beam_size, 5)

                    logger.info(f"🎤 Transcribing with Whisper ({whisper_model_size} model, tier={tier}, beam_size={current_beam_size})...")
                    try:
                        # Use pre-decoded audio_np if available, otherwise fallback to input_data
                        source = audio_np if audio_np is not None else (io.BytesIO(input_data) if isinstance(input_data, bytes) else input_data)
                        
                        # Pass language preference to Whisper if specified
                        transcribe_kwargs = {
                            "beam_size": current_beam_size,
                            "vad_filter": True,
                            "vad_parameters": dict(min_silence_duration_ms=2000, speech_pad_ms=500), # Allow 2s silence for blocks, add padding to catch start/end
                            "word_timestamps": False,
                            "condition_on_previous_text": False, # Prevents looping on stutters
                            "patience": 2.0, # Explore more decoding paths (Critical for Telugu accuracy)
                            "no_speech_threshold": 0.3, # Highly sensitive to speech (default 0.6)
                            "log_prob_threshold": None # Do not discard low confidence speech (Hear everything)
                        }
                        if lang_pref != "auto":
                            transcribe_kwargs["language"] = lang_pref
                            # Disable temperature fallback to prevent switching languages on low confidence
                            transcribe_kwargs["temperature"] = 0.0
                            # Add initial prompt to guide the model towards the correct script and context
                            if lang_pref in INITIAL_PROMPTS:
                                transcribe_kwargs["initial_prompt"] = INITIAL_PROMPTS[lang_pref]

                        segments_gen, info = whisper_model.transcribe(source, **transcribe_kwargs)
                        return list(segments_gen), info
                    except Exception as e:
                        logger.error(f"❌ Whisper Transcription Error: {e}")
                        return [], None
                
                segments, info = await loop.run_in_executor(None, lambda: transcribe_blocking(attempt_tier))
                
                if info is None:
                    break
                
                current_text = "".join([s.text for s in segments]).strip()
                
                # Filter Whisper hallucinations
                hallucinations = ["[Silence]", "[Music]", "[BLANK_AUDIO]", "(silence)", "(music)"]
                if current_text.strip() in hallucinations:
                    current_text = ""
                
                if current_text:
                    transcribed_text = current_text
                    detected_lang = info.language
                    break
                else:
                    if attempt_tier != tiers_to_try[-1]:
                        logger.warning(f"⚠️ Empty result with {attempt_tier} tier. Retrying with next tier...")
            
            if info is None:
                raise ValueError("Audio transcription failed. The audio file might be corrupt or unsupported.")

        # Handle empty transcription
        if not transcribed_text:
            logger.warning("⚠️ Whisper produced an empty transcription. Stopping pipeline.")
            return {
                "status": "error",
                "message": "No speech was detected in the audio. Please try recording again.",
                "input_audio_url": input_audio_url,
                "input_text": "(No speech detected)",
                "output_audio_url": None,
                "output_text": "",
                "analysis": {},
                "response": {},
                "metrics": {"wer": 1.0},
                "detected_language": detected_lang,
                "transcription": "(No speech detected)"
            }

        # Calculate dynamic speed based on articulation rate (ignoring stutter blocks)
        input_wpm = 150 # Default
        if not is_text_input:
            speech_duration = 0
            word_count = 0
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        speech_duration += (word.end - word.start)
                        word_count += 1
            
            effective_duration = speech_duration if speech_duration > 0.5 else info.duration
            input_wpm = (word_count / effective_duration) * 60 if effective_duration > 0 else 150
        
        if speed_pref:
            final_speed = float(speed_pref)
        else:
            final_speed = max(0.8, min(1.3, input_wpm / 150)) # Clamp speed to natural range

        # Improved Auto-Detection Logic
        if lang_pref == "auto":
            try:
                # Creative Heuristic: Check for Latin Script dominance
                # If Whisper detects a non-Latin language (e.g., Hindi, Telugu) but the text is mostly English letters,
                # it is likely English spoken with an accent.
                
                alpha_chars = [c for c in transcribed_text if c.isalpha()]
                latin_ratio = 0.0
                if alpha_chars:
                    latin_count = sum(1 for c in alpha_chars if 'a' <= c.lower() <= 'z')
                    latin_ratio = latin_count / len(alpha_chars)

                # Languages that typically use non-Latin scripts
                non_latin_langs = ['hi', 'te', 'bn', 'kn', 'ta', 'ml', 'zh', 'ja', 'ru']
                # Indian languages often use Latin script (transliteration)
                transliteration_friendly = ['hi', 'te', 'bn', 'kn', 'ta', 'ml']
                
                text_lang = detect(transcribed_text)

                logger.info(f"🔍 Auto-Detect Debug: Whisper={detected_lang}, TextDetect={text_lang}, LatinRatio={latin_ratio:.2f}")

                # Logic update: Only default to English if it's NOT a transliteration-friendly language
                if (detected_lang in non_latin_langs and detected_lang not in transliteration_friendly and latin_ratio > 0.5):
                     logger.info(f"⚠️ Language conflict detected (Script mismatch). Defaulting to English.")
                     final_lang = 'en'
                elif (text_lang == 'en' and detected_lang != 'en' and detected_lang not in transliteration_friendly):
                     # If text detection says English strongly, and Whisper disagrees, but it's not a transliterated lang
                     final_lang = 'en'
                else:
                     final_lang = detected_lang
            except Exception as e:
                logger.warning(f"⚠️ Auto-detection heuristic failed: {e}")
                final_lang = detected_lang
        else:
            final_lang = lang_pref

        # Map code to full name for Agent (e.g., "te" -> "Telugu") for better LLM accuracy
        agent_lang_name = LANGUAGE_NAMES.get(final_lang, final_lang)

        # 2. AI Analysis
        # Run blocking AI call in a separate thread to prevent server freeze
        ai_data = await loop.run_in_executor(None, lambda: knowledge_agent_client(transcribed_text, language=agent_lang_name, tone=tone_pref, session_id=session_id, acoustic_features=acoustic_features))
        
        if not ai_data:
            # Fallback for connection error
            ai_data = {
                "text": transcribed_text,
                "english_translation": "N/A",
                "analysis": "<ul><li><strong>Processing Error</strong>: The AI response could not be parsed.</li><li>Please check the server logs for the raw output.</li></ul>",
                "suggestions": "<ul><li>Try recording again with clearer speech.</li></ul>",
                "metrics": {"words": len(transcribed_text.split()) if transcribed_text else 0, "disfluencies": 0, "rate": 100},
                "soap": {"s": "N/A", "o": "N/A", "a": "Processing Error", "p": "Retry"},
                "level": "Error",
                "classification": "Parsing Failure",
                "demographics": "Unknown"
            }

        # 3. TTS
        corrected_text = ai_data.get("text", "")
        output_audio_url = await tts_router(corrected_text, final_lang, voice_pref, speed=final_speed)
        
        # 4. Calculate Metrics (WER / Correction Rate)
        # We treat the "Corrected Text" as the Reference (Ground Truth) and Input as Hypothesis
        wer_score = 0.0
        if corrected_text and transcribed_text:
            try:
                wer_score = jiwer.wer(corrected_text, transcribed_text)
            except:
                wer_score = 0.0

        return {
            "input_audio_url": input_audio_url,
            "input_text": transcribed_text,
            "output_audio_url": output_audio_url,
            "output_text": corrected_text,
            "analysis": ai_data,
            "response": {**ai_data, "audio_url": output_audio_url, "input_audio_url": input_audio_url, "acoustic_features": acoustic_features}, # For new UI compatibility
            "metrics": {"wer": wer_score},
            "detected_language": detected_lang,
            "status": "success",
            "transcription": transcribed_text
        }
    except Exception as e:
        logger.error("❌ Pipeline Error:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "input_audio_url": input_audio_url,
            "transcription": "(Error)"
        }

@app.get("/voices")
def get_voices():
    logger.info("🎤 Serving voice list...")
    # Prevent browser caching of the voice list to ensure new voices appear immediately
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}

    return JSONResponse(content={
        "en": [
            {"id": "af_heart", "name": "Heart (Female - US - Premium)"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia (Female - UK)"},
            {"id": "en-GB-RyanNeural", "name": "Ryan (Male - UK)"},
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (Female - Indian Accent)"},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Male - Indian Accent)"},
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Male - Hindi Accent for English)"},
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Female - Hindi Accent for English)"},
            {"id": "te-IN-MohanNeural", "name": "Mohan (Male - Telugu Accent for English)"},
            {"id": "te-IN-ShrutiNeural", "name": "Shruti (Female - Telugu Accent for English)"}
        ],
        "hi": [
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Male)"}
        ],
        "te": [
            {"id": "te-IN-MohanNeural", "name": "Mohan (Male)"}
        ],
        "bn": [
            {"id": "bn-IN-BashkarNeural", "name": "Bashkar (Male)"}
        ],
        "kn": [
            {"id": "kn-IN-GaganNeural", "name": "Gagan (Male)"}
        ],
        "ml": [
            {"id": "ml-IN-MidhunNeural", "name": "Midhun (Male)"}
        ],
        "ta": [
            {"id": "ta-IN-ValluvarNeural", "name": "Valluvar (Male)"}
        ],
        "es": [
            {"id": "es-ES-AlvaroNeural", "name": "Alvaro (Male - Spain)"},
            {"id": "es-MX-JorgeNeural", "name": "Jorge (Male - Mexico)"}
        ],
        "fr": [
            {"id": "fr-FR-HenriNeural", "name": "Henri (Male - France)"},
            {"id": "fr-CA-AntoineNeural", "name": "Antoine (Male - Canada)"}
        ],
        "de": [
            {"id": "de-DE-KillianNeural", "name": "Killian (Male)"}
        ],
        "zh": [
            {"id": "zh-CN-YunxiNeural", "name": "Yunxi (Male)"}
        ],
        "ja": [
            {"id": "ja-JP-KeitaNeural", "name": "Keita (Male)"}
        ],
        "ru": [
            {"id": "ru-RU-DmitryNeural", "name": "Dmitry (Male)"}
        ]
    }, headers=headers)

def main():
    app_port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Fluency-Net ({ACTIVE_TIER} tier)...")
    print(f"ℹ️  Voice Profile: {VOICE_PROFILE}, Speed: {SPEED}x")
    print(f"👉 Open your browser at http://localhost:{app_port}")
    print(f"   (Set the 'PORT' environment variable to use a different port if {app_port} is busy)")
    # webbrowser.open(f"http://localhost:{app_port}") # Disable auto-open for Docker/Server envs
    uvicorn.run(app, host="0.0.0.0", port=app_port)

if __name__ == "__main__":
    main()
