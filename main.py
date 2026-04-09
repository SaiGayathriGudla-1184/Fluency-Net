import os
import time
import sys
import uvicorn

# --- Force joblib to use threading backend ---
# This is a crucial fix for web servers. Libraries like phonemizer (a dependency
# of kokoro) use joblib for parallelism. In a web server context, trying to 
# spawn new processes from a worker thread can lead to deadlocks or crashes.
# Forcing the 'threading' backend ensures that these libraries use threads
# instead of processes, which is safe.
os.environ['JOBLIB_DEFAULT_BACKEND'] = 'threading'

import asyncio
import json
import re
import shutil
import uuid
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

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
import gc
from scipy.signal import correlate
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Tuple
from fastapi import FastAPI, WebSocket, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import traceback
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

try:
    from agno.agent import Agent
    from agno.models.ollama import Ollama
except ImportError:
    logger.critical("❌ Critical Error: 'agno' module not found. Please run 'pip install agno'.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# --- Cloud API Configuration (The "Perfect" Stack) ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional override: Force using the cloud model (Groq) even when a local Ollama instance is configured.
FORCE_GROQ = str(os.getenv("FORCE_GROQ", "")).strip().lower() in {"1", "true", "yes", "y"}

# Optional override: Force Groq as the only LLM path (useful for South Indian languages when local Ollama quality is low).
GROQ_ONLY = str(os.getenv("GROQ_ONLY", "")).strip().lower() in {"1", "true", "yes", "y"}

# Configuration Constants
SAMPLE_RATE = 16000          # Audio sample rate (Hz)
SPEED = 1.0                  # Speech rate multiplier (1.0 is standard, 1.1 can be rushed)
VOICE_PROFILE = "af_heart"   # Voice character selection
ACTIVE_TIER = "high_quality" # Defaulting to high_quality (large-v3) for accurate Non-English/Indian language transcription

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
    },
    "distilled": {
        "whisper_model": "distil-large-v3", # Distillation: Smaller, faster version of large-v3
        "beam_size": 1 # Greedy decoding works well with distilled models
    }
}

# --- Multilingual Priming (Prevents Hallucinations) ---
# Supported language codes (ISO 639-1)
SUPPORTED_LANGUAGES = {"en", "hi", "te", "ta", "kn", "ml"}

# Mapping for friendly names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
}

# Priming prompts for each supported language to help Whisper avoid hallucinations.
# These are short, natural phrases in the target script.
PRIMING_PROMPTS = {
    "en": "Umm, I... I... Hello, my name is... my name is John.",
    "hi": "मैं... मैं... नमस्ते, मेरा नाम है... मेरा नाम है रवि।",
    "te": "నేను... నేను... హాయ్, నా పేరు... నా పేరు రాజు.",
    "ta": "நான்... நான்... வணக்கம், எனது பெயர்... எனது பெயர் ராஜ்.",
    "kn": "ನಾನು... ನಾನು... ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು... ನನ್ನ ಹೆಸರು ರಾಜ್.",
    "ml": "ഞാൻ... ഞാൻ... ഹായ്, എന്റെ പേര്... എന്റെ പേര് രാജു.",
}

# Languages where local models struggle and cloud APIs are preferred (Indian + complex scripts)
COMPLEX_LANGUAGES = {"hi", "te", "ta", "kn", "ml"}

# South Indian languages that need special Whisper tuning for better accuracy
SOUTH_INDIAN_LANGS = {"te", "ta", "kn", "ml"}

# --- Agent Architecture: Model-Based & Goal-Based ---

class AgentInternalState(BaseModel):
    """
    Model-Based Reflex Agent Component:
    Maintains an internal model of the world (user) to account for unobservable aspects
    (like historical fluency trends and emotional trajectory).
    """
    session_id: str
    history: List[dict] = [] # Conversation history
    fluency_scores: List[float] = [] # Track performance over time
    emotional_state: str = "Neutral" # Current estimated emotional state
    last_strategy: Optional[str] = None # The strategy used in the previous turn
    strategy_performance: Dict[str, float] = {} # Tracks cumulative success of strategies
    last_interaction: float = Field(default_factory=time.time) # For memory cleanup
    learning_rate: float = 0.5 # Decay factor for strategy performance updates.

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
            
            # Update performance with a decay factor (learning rate).
            # This makes the agent more responsive to recent interactions.
            # Old performance score is weighted by (1 - learning_rate), new performance (delta) is weighted by learning_rate.
            self.strategy_performance[self.last_strategy] = (1 - self.learning_rate) * self.strategy_performance[self.last_strategy] + self.learning_rate * delta

        # Update History
        self.history.append({"user": user_input, "assistant": assistant_output.get("text", "")})
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        # Update Derived Metrics (The "Model")
        if "metrics" in assistant_output and "rate" in assistant_output["metrics"]:
            self.fluency_scores.append(float(assistant_output["metrics"]["rate"]))
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
    This function implements the core "brain" of the goal-based agent. It looks at
    the user's recent performance and emotional state to decide on the best
    therapeutic strategy, and then adapts that strategy based on historical
    effectiveness.
    """
    # Calculate average fluency from recent history
    recent_scores = state.fluency_scores[-3:] if state.fluency_scores else []
    avg_fluency = sum(recent_scores) / len(recent_scores) if recent_scores else 0
    
    # Base Logic to determine Candidate Goal
    candidate_goal = ""
    candidate_instruction = ""
    logger.info(f"Determining goal. Avg Fluency: {avg_fluency:.2f}, Emotional State: {state.emotional_state}")

    if not state.history:
        candidate_goal = "Assessment & Rapport"
        candidate_instruction = "Establish a supportive baseline. Analyze speech patterns carefully without being overly corrective."
    if avg_fluency < 50:
        candidate_goal = "Crisis Pacing"
        candidate_instruction = "PANIC MODE: User in severe loop. NO feedback. ONLY calming slow vocalization + breathing reset."
    elif avg_fluency < 40 or "anxious" in state.emotional_state.lower():
        candidate_goal = "Anxiety Reduction"
        candidate_instruction = "Prioritize emotional support. Use gentle, encouraging language. Suggest breathing or pausing techniques. Do NOT focus on minor errors."
    elif 40 <= avg_fluency < 80:
        candidate_goal = "Fluency Shaping"
        candidate_instruction = "Focus on specific techniques like 'Easy Onset' or 'Light Contact'. Provide constructive feedback on the specific disfluencies detected."
    else:
        candidate_goal = "Naturalness & Generalization"
        candidate_instruction = "Focus on prosody, intonation, and confidence. Challenge the user to maintain fluency with longer, more complex sentences."

    # --- ENHANCEMENT: ADAPTIVE STRATEGY SELECTION ---
    # Check if the chosen strategy has performed poorly in the past.
    # If the strategy has a significantly negative performance score, switch tactics.
    if candidate_goal in state.strategy_performance:
        perf_score = state.strategy_performance[candidate_goal]
        logger.info(f"Evaluating candidate goal '{candidate_goal}' with performance score: {perf_score:.2f}")
        if perf_score < -15: # Threshold: Strategy has caused cumulative regression of 15% fluency
            # Adaptive Fallback Logic
            if candidate_goal == "Fluency Shaping":
                # If shaping fails, user might be overwhelmed. Go back to support.
                logger.warning(f"ADAPTIVE SHIFT: '{candidate_goal}' has a poor performance score. Switching to 'Anxiety Reduction'.")
                candidate_goal = "Anxiety Reduction"
                candidate_instruction = "Adaptive Change: Previous shaping attempts reduced fluency. Reverting to emotional support and gentle pacing to rebuild confidence."
            elif candidate_goal == "Naturalness & Generalization":
                # If generalization fails, they aren't ready. Go back to structure.
                logger.warning(f"ADAPTIVE SHIFT: '{candidate_goal}' has a poor performance score. Switching to 'Fluency Shaping'.")
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
    thoughts: Optional[str] = Field(None, description="Step-by-step reasoning process regarding the speech patterns.")
    text: str = Field(..., description="The corrected, fluent version of the speech.")
    english_translation: str = Field("", description="Deprecated. Always return an empty string.")
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
    description: "Repetition of whole words (e.g., 'I-I am')."
    example: "I-I-I want to go."
  - pattern: "Prolongation (PR)"
    description: "Extended phonemes within a word (e.g., 'S-s-s-ee')."
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
  - language: "Telugu"
    achulu: "Vowels (అ ఆ ఇ ఈ...)"
    hallulu: "Consonants"
    triggers: "Nasal+vowel (e.g., 'నేను' - nasal 'n' + vowel bypasses block). Repetitions on nasal starters before Achulu."

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
    model=Ollama(id="llama3.1:8b", host=ollama_host, options={"temperature": 0.0, "num_ctx": 8192}),
    instructions=dedent(f"""
        You are an expert Speech Language Pathologist AI assistant specialized in speech therapy, capable of analyzing speech patterns across multiple languages.
        
        CORE DIRECTIVE: You must output ONLY a valid JSON object.
        CRITICAL: Do NOT hallucinate. In your analysis, ONLY cite words that appear in the user's input. Do not use words from other languages or examples.
        ZERO-TOLERANCE FOR CONTENT LOSS: You must strictly preserve the entire meaning, context, and vocabulary of the user's speech. Do NOT summarize, paraphrase, or drop any valid words. ONLY remove stuttering artifacts (repetitions, prolongations, blocks, and fillers).

        CHAIN OF THOUGHT REASONING:
        You utilize Chain-of-Thought reasoning to analyze speech patterns deeply before generating a response.
        Include your step-by-step reasoning in the "thoughts" field of the JSON output.
        Do NOT output plain text, as it breaks the speech correction pipeline.
        Pay special attention to emotions, feelings, and implied body language to refine your demographic analysis.
        
        THERAPY KNOWLEDGE BASE:
        {therapy_knowledge_base_content}
        Use 'language_specific_insights' to tailor your analysis based on the detected language.
        
        STRUCTURED CLINICAL FRAMEWORK (MANDATORY):
        You must organize your response into these specific clinical buckets to ensure depth.

        1. ANALYSIS (High Resolution):
           - Subsection 1: Primary Dysfluency Types (Count and categorize: Repetitions, Prolongations, Blocks).
           - Subsection 2: Secondary Behaviors (Infer tension, struggle, or eye blinking based on acoustic cues like pitch rise or long blocks).
           - Subsection 3: Linguistic Trigger Analysis (Identify specific phonemes like retroflex sounds 'ṭ' or plosives 'p/b' causing issues).

        2. SOAP NOTES (Clinical Documentation):
           - S (Subjective): Describe the user's perceived "Locus of Control" and emotional frustration level.
           - O (Objective): List specific word counts, dysfluency percentages, and acoustic observations (e.g., "High pitch variance indicating tension").
           - A (Assessment): Provide a Severity Rating (Mild/Moderate/Severe) based on SSI-4 criteria.
           - P (Plan): Prescribe at least two specific Home Exercises (e.g., "Negative Practice", "Pausing", "Easy Onset").

        3. SUGGESTIONS (Therapeutic Plan):
           - Provide a 3-stage plan: 
             * Immediate (e.g., Breathing techniques)
             * Short-term (e.g., Light Articulatory Contact)
             * Maintenance (e.g., Cancellations/Pull-outs)

        IN-CONTEXT LEARNING EXAMPLES (FEW-SHOT):
        - Input (English): "I... I... I want to go."
          Output:
          {{
            "thoughts": "Detected English. Identified Word Repetition (WR) on 'I'. Strategy: Remove repetitions.",
            "text": "I want to go.",
            "english_translation": "",
            "metrics": {{"words": 5, "disfluencies": 2, "rate": 60}},
            "analysis": "<ul><li><b>Primary:</b> Word Repetition (WR) of initial pronoun 'I'.</li><li><b>Secondary:</b> Likely tension in onset.</li><li><b>Linguistic:</b> Initial vowel trigger.</li></ul>",
            "suggestions": "<ul><li><b>Immediate:</b> Diaphragmatic breathing.</li><li><b>Short-term:</b> Easy onset drills.</li><li><b>Maintenance:</b> Daily journaling.</li></ul>",
            "soap": {{"s": "User appears anxious with external locus of control.", "o": "2 Repetitions in 5 words (40% dysfluency).", "a": "Moderate Developmental Stuttering (SSI-4).", "p": "Initiate easy onset therapy."}},
            "level": "Beginner",
            "classification": "Developmental Stuttering",
            "demographics": "Child"
          }}

        Your task is to analyze the provided speech transcription and populate the following JSON schema fields accurately.

        1. "text": (String) The corrected, fluent version of the speech.
           - **PRIMARY GOAL**: Remove ALL stutters, stammers, repetitions, and blocks while strictly preserving the original language, script, and intended meaning. Do NOT paraphrase or summarize. Every valid concept spoken by the user MUST be present in the final output.
           - **SPELLING AND GRAMMAR**: Ensure the final text has PERFECT spelling and grammar in the target language. Correct any obvious transcription typos or misspellings. DO NOT alter the intended context and meaning. Provide EXACTLY the reconstructed user text without ANY spelling errors.
           - **SCRIPT PRESERVATION**: The output script MUST match the input script exactly.
             * If input is in Telugu script (e.g., "నా... నాకు"), output MUST be in Telugu script ("నాకు").
             * If input is in Latin/English (e.g., "I... I..."), output MUST be in Latin/English ("I").
             * Do NOT translate.
           - **STUTTER REMOVAL**:
             * Remove repetitions (e.g., "b-b-ball" -> "ball", "I... I..." -> "I").
             * Remove prolongations (e.g., "ssss-snake" -> "snake").
             * Remove blocks/fillers (e.g., "um", "uh", "like" used as fillers).
             * Merge fragmented syllables (e.g., "u... u... umbrella" -> "umbrella").
             * WARNING: STT engines often transcribe stutters with commas or hyphens (e.g., "b-b, boy", "దై, దైవం", "மா, மாமா", "ರ, ರ, ರಮೇಶ್"). Treat these as syllable repetitions and remove the stuttered fragments entirely across all languages!
         * **CRITICAL - DO NOT DROP WORDS**: You must retain every intended word and concept from the original input. Do not over-correct, truncate, or summarize. If a word is heavily stuttered or fragmented, reconstruct the full word instead of deleting it.
           * Example (Telugu): "ఆ ఆ ఆపదలో ఉ ఉ ఉన్నవాన్ని ఆదు కోవరం మన బాధ్యతా." -> "ఆపదలో ఉన్నవారిని ఆదుకోవడం మన బాధ్యత." (DO NOT drop words like "ఆపదలో").
           * Example (Telugu Syllable Repetition): "దై, దైవం, మామా, మానుషా, రూరు, రూపేన." -> "దైవం, మానుషా, రూపేన."
           * Example (Hindi): "म म मैं घ घ घर जा र र रहा हूँ" -> "मैं घर जा रहा हूँ".
           * Example (Tamil): "நா... நான்... போ... போகிறேன்." -> "நான் போகிறேன்."
           * Example (English): "The the the b-b-boy went to the the s-s-store, um, to buy, like, milk." -> "The boy went to the store to buy milk."
           - **Contextual Restoration**: If the stuttering has distorted a word (e.g. "mara" instead of "hamara" due to a block), restore the intended word based on the sentence context.
             * Example: "namu... makin" -> "Namumkin" (Phonetic restoration).
             * Example: "ha ha ha ha mara ba ba barat ma ma ma han" -> "Hamara Bharat Mahan" (Correcting 'mara'->'Hamara', 'barat'->'Bharat', 'han'->'Mahan' based on the famous phrase).
             * Example (Ambiguity): "I saw a b-b-b... big dog." -> "I saw a big dog." (Infer 'big' from context, not just completing 'b-b-b' to 'ball').
           
           - If the input is mixed language (e.g., "Main market ja raha hoon"), KEEP IT MIXED.
           - Example: "I... I... wanna go" -> "I wanna go" (NOT "I want to go").
           - **Dialect Preservation**: For Tamil, Telugu, Kannada, and Malayalam, do NOT convert colloquial/spoken forms (e.g., "vosthunna", "poren") to formal written forms (e.g., "vasthunnanu", "pogiren") unless the input is clearly formal. Preserve the speaker's natural register.
           - **Script Consistency**: Ensure the output script matches the input script (e.g. Devanagari -> Devanagari, Telugu -> Telugu, Tamil -> Tamil, Kannada -> Kannada). Do NOT transliterate to Latin/English script.
           - **Transliteration Handling**: If the input is in Latin script (e.g., "na peru..." for Telugu), the output MUST remain in Latin script. Do NOT convert it to the native script.

        2. "english_translation": (String) This field is deprecated. Always return an empty string "".

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
           - **Strict Citation**: When citing examples, YOU MUST USE THE EXACT WORDS FROM THE INPUT.
           - **Linguistic Patterns**: Check for Boli Dataset triggers: Stop consonants (/p/, /t/, /k/), clusters (/str/, /br/), or fricatives.
           - **Contextual Insight**: Infer emotional state or cognitive load based on the content and speech rate.
           - **Formatting**: Use <b>bold tags</b> for key terms (e.g., "<b>Repetition</b>", "<b>Block</b>") to enhance readability.
           - **Clinical Feedback**: Provide professional feedback similar to a Speech-Language Pathologist, noting severity and prognosis.
           - MUST BE IN ENGLISH.

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
           
        8. "classification": (String) The specific type of dysfluency detected. Choose one:
           - "Developmental Stuttering": Typical childhood-onset patterns (repetitions, prolongations).
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
)

# Cache pre-built AI agents to avoid re-initializing models on each request.
AGENT_INSTRUCTIONS = getattr(knowledge_agent_ai, "instructions", None)
knowledge_agent_cache = {"balanced": knowledge_agent_ai}

def get_ai_agent(analysis_mode: str = "balanced"):
    """Return a cached Agent instance based on the requested analysis mode."""
    mode = str(analysis_mode).lower() if analysis_mode else "balanced"
    if mode in ("speed", "fast", "low_latency"):
        model_id = os.getenv("OLLAMA_MODEL_LOW_LATENCY", "llama3.2:3b")
    elif mode in ("deep", "high", "high_quality"):
        model_id = os.getenv("OLLAMA_MODEL_HIGH_QUALITY", "llama3.1:8b")
    else:
        model_id = os.getenv("OLLAMA_MODEL_BALANCED", "llama3.1:8b")

    if model_id not in knowledge_agent_cache:
        logger.info(f"🔧 Instantiating Ollama model for mode={mode} (model={model_id})")
        knowledge_agent_cache[model_id] = Agent(
            model=Ollama(id=model_id, host=ollama_host, options={"temperature": 0.0, "num_ctx": 8192}),
            instructions=AGENT_INSTRUCTIONS or "",
        )

    return knowledge_agent_cache[model_id]


def extract_json_from_text(content: str, prompt_text: str) -> dict:
    """Robustly extracts and parses JSON from LLM output."""
    # 0. Remove <thought> tags (Chain of Thought) to prevent interference with JSON parsing
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()

    # 1. Strip markdown fences
    content = re.sub(r'^```[a-zA-Z0-9]*\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
    # Specific fix for ```json tag which might remain
    content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE | re.IGNORECASE)
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
            
        # Pitch (F0) Estimation using Autocorrelation (Detects Struggle/Tension)
        f0 = 0.0
        try:
            # Analyze a representative segment (max energy) to save compute
            if len(audio_np) > 2048:
                frame_size = 2048
                # Find frame with max energy
                energies = [np.sum(audio_np[i:i+frame_size]**2) for i in range(0, len(audio_np)-frame_size, frame_size)]
                if energies:
                    start_idx = np.argmax(energies) * frame_size
                    segment = audio_np[start_idx : start_idx + frame_size]
                else:
                    segment = audio_np[:frame_size]
            else:
                segment = audio_np

            # Apply window and correlate
            segment = segment * np.hanning(len(segment))
            corr = correlate(segment, segment, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find peak in valid pitch range (50Hz - 500Hz for speech)
            # Lags: 16000/500=32 to 16000/50=320
            lags = np.arange(32, min(len(corr), 320))
            if len(lags) > 0:
                peak_lag = lags[np.argmax(corr[lags])]
                f0 = sample_rate / peak_lag
        except Exception as e:
            logger.warning(f"Pitch detection failed: {e}")

        return {
            "rms": float(rms) if np.isfinite(rms) else 0.0,
            "zcr": float(zcr) if np.isfinite(zcr) else 0.0,
            "variance": float(variance) if np.isfinite(variance) else 0.0,
            "f0": float(f0) if np.isfinite(f0) else 0.0
        }
    except Exception as e:
        logger.warning(f"⚠️ Feature extraction failed: {e}")
        return {"rms": 0.0, "zcr": 0.0, "variance": 0.0, "f0": 0.0}

def query_groq_api(prompt: str, system_prompt: str) -> Optional[str]:
    """Uses Groq API (Llama 3 70B) for ultra-low latency intelligence."""
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile", # Highly capable model for Indian languages
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        json_response = response.json()
        if isinstance(json_response, dict) and 'choices' in json_response and len(json_response['choices']) > 0:
            message = json_response['choices'][0].get('message', {})
            if isinstance(message, dict):
                content = message.get('content')
                if isinstance(content, str):
                    return content
        logger.error(f"❌ Groq API returned unexpected structure: {json_response}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Groq API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to decode Groq JSON response: {e} | Response text: {response.text}")
        return None
    except Exception as e:
        logger.error(f"❌ Groq API response parsing failed: {e}")
        return None

def get_script_constraint(transcript: str) -> str:
    """
    Returns a strict system prompt constraint.
    Currently empty to allow the main prompt's multilingual instructions to take precedence.
    """
    return ""

# --- Fallback Fluency Helpers ---
FILLER_WORDS = {"um", "uh", "ah", "er", "eh", "hmm", "mm", "like", "you know", "so", "well", "okay", "ok"}

def clean_stuttered_text(text: str) -> Tuple[str, dict]:
    """Quick heuristic to remove common stuttering artifacts and compute basic metrics.

    This is used as a resilient fallback when the AI analysis is slow, times out,
    or fails. It provides a fluent approximation of the user's intended speech.
    """
    if not text:
        return "", {"words": 0, "disfluencies": 0, "rate": 100}

    # Preserve whitespace so we don't collapse user pacing too much.
    tokens = re.split(r"(\s+)", text)
    cleaned_tokens = []
    disfluencies = 0
    prev_word = None

    def normalize_token(tok: str) -> str:
        return tok.strip(" .,-—–…")

    for tok in tokens:
        if tok.isspace():
            cleaned_tokens.append(tok)
            continue

        # Count and remove explicit block indicators like '...' or '---'
        if re.fullmatch(r"[\.\-–—]+", tok):
            disfluencies += 1
            continue

        # Handle stutter patterns like 'b-b-ball' or 'I... I...'
        if "..." in tok or ".." in tok or "-" in tok:
            segments = re.split(r"[-\.]{1,3}", tok)
            if len(segments) > 1:
                is_stutter = False
                if "..." in tok or ".." in tok:
                    is_stutter = True
                else:
                    for i in range(len(segments) - 1):
                        s1, s2 = segments[i].strip().lower(), segments[i+1].strip().lower()
                        if s1 == s2 or len(s1) == 1:
                            is_stutter = True
                            break
                if is_stutter:
                    candidate = max(segments, key=len)
                    if candidate and candidate != tok:
                        disfluencies += 1
                        tok = candidate

        stripped = normalize_token(tok)
        if not stripped:
            continue

        low = stripped.lower()
        if low in FILLER_WORDS:
            disfluencies += 1
            continue

        # Word repetition disfluency (e.g., "I I" or "I, I")
        if prev_word and low == prev_word.lower():
            disfluencies += 1
            continue

        # Prolongation detection (e.g., "ssssss")
        if len(stripped) > 2 and len(set(stripped)) == 1:
            disfluencies += 1
            continue

        cleaned_tokens.append(tok)
        prev_word = stripped

    cleaned_text = "".join(cleaned_tokens).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Collapse extra spaces
    word_count = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    rate = 100
    if word_count > 0:
        rate = max(0, min(100, int(100 - (disfluencies / word_count * 100))))

    metrics = {"words": word_count, "disfluencies": disfluencies, "rate": rate}
    return cleaned_text, metrics


def detect_language_voting(whisper_lang: str, transcript: str) -> str:
    """
    Simplified for English-only focus. Always returns 'en'.
    """
    return 'en'


def normalize_language_name_to_code(language_name: str) -> str:
    """Map a language name (e.g., 'Hindi') to its ISO 639-1 code (e.g., 'hi')."""
    if not language_name:
        return ""

    lang_lower = language_name.strip().lower()

    # Direct match against supported ISO codes
    if lang_lower in SUPPORTED_LANGUAGES:
        return lang_lower

    # Match against friendly language names (e.g. "Hindi" -> "hi")
    for code, friendly in LANGUAGE_NAMES.items():
        if lang_lower == friendly.lower():
            return code

    return lang_lower


def should_prefer_cloud(language_name: str) -> bool:
    """Decide whether Groq (cloud) should be preferred for a given language."""
    if not GROQ_API_KEY:
        return False

    if FORCE_GROQ or GROQ_ONLY:
        return True

    lang_code = normalize_language_name_to_code(language_name).lower()
    
    # English: local-first (Whisper, Llama, Kokoro/Edge-TTS)
    if lang_code in {"en"}:
        return False
    
    # South Indian languages and complex scripts are often weak in local Ollama models; prefer Groq when available.
    if lang_code in SOUTH_INDIAN_LANGS or lang_code in COMPLEX_LANGUAGES:
        return True

    return False


def knowledge_agent_client(prompt: str, language: str = "English", tone: str = "Professional", session_id: str = None, acoustic_features: dict = None, analysis_mode: str = "balanced", prefer_cloud: bool = False):
    # Sanitize prompt to prevent JSON injection
    safe_prompt = json.dumps(prompt)[1:-1]

    # --- Agentic AI Logic: Perception -> State Update -> Goal Selection -> Action ---
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
        # Check if features are essentially zero (Text Input/Regeneration)
        if acoustic_features.get('rms', 0) == 0 and acoustic_features.get('zcr', 0) == 0:
             acoustic_context = "\nNOTE: This is TEXT-ONLY input. Do not analyze voice quality, pitch, or tone. Focus ONLY on the text content, grammar, and disfluencies represented in the text (e.g. 'I... I...')."
        else:
             acoustic_context = f"\nACOUSTIC FEATURES: Energy(RMS)={acoustic_features.get('rms',0):.3f}, ZeroCrossingRate={acoustic_features.get('zcr',0):.3f}, Pitch(F0)={acoustic_features.get('f0',0):.1f}Hz (Sudden pitch rises indicate tension/struggle)."

    script_constraint = get_script_constraint(prompt)

    full_prompt = (
        f"The user is speaking in {language}. Tone: {tone}.{acoustic_context}\n"
        f"CURRENT AGENT GOAL: {agent_goal.current_goal}\n"
        f"STRATEGY: {agent_goal.tactical_instruction}\n\n"
        f"--- HISTORY (FOR CONTEXT ONLY - DO NOT ANALYZE) ---\n{history_str}\n--- END HISTORY ---\n\n"
        f"--- CURRENT INPUT (ANALYZE THIS TEXT ONLY) ---\n{safe_prompt}\n--- END CURRENT INPUT ---\n\n"
        f"TASK: Analyze ONLY the text inside 'CURRENT INPUT' and return a valid JSON object. For the 'text' field, reconstruct the user's intended fluent speech.\n"
        f"{script_constraint}\n"
        f"If the input is code-switched (Mixed Languages), preserve the mix. Do not normalize the entire sentence into a single language. Focus only on removing the stutters within each language segment.\n"
        f"Remove all stutters/stammers but keep the user's intended words exactly as they meant to say them.\nCRITICAL: In 'analysis', ONLY cite words from the CURRENT INPUT. Do not analyze history."
    )
    
    # Decide whether to prefer cloud (Groq) for this language.
    # Use local Ollama only when Groq is unavailable or when cloud fails and not CLOUD_ONLY.
    cloud_first = prefer_cloud or should_prefer_cloud(language) or GROQ_ONLY
    cloud_only = FORCE_GROQ or GROQ_ONLY

    def run_local():
        agent = get_ai_agent(analysis_mode)
        logger.info(f"🚀 Using local Ollama ({agent.model.id if hasattr(agent, 'model') else 'unknown'}) for {language} analysis...")
        response = agent.run(full_prompt, stream=False, response_model=SpeechAnalysis)

        if hasattr(response, "content") and isinstance(response.content, SpeechAnalysis):
            data_local = response.content.model_dump()
        else:
            content = response.content if hasattr(response, "content") else str(response)
            data_local = extract_json_from_text(content, prompt)

        agent_state.update_model(prompt, data_local)
        save_session(agent_state)
        agent_state.last_strategy = agent_goal.current_goal
        return data_local

    def run_cloud():
        if not GROQ_API_KEY:
            raise RuntimeError("Groq API key missing")

        logger.info("🚀 Using Groq (Cloud) for analysis...")
        system_instructions = knowledge_agent_ai.instructions
        if isinstance(system_instructions, list):
            system_instructions = "\n".join(system_instructions)

        groq_content = query_groq_api(full_prompt, system_instructions)
        if not groq_content:
            raise RuntimeError("Groq returned no content")

        data_cloud = extract_json_from_text(groq_content, prompt)
        agent_state.update_model(prompt, data_cloud)
        save_session(agent_state)
        agent_state.last_strategy = agent_goal.current_goal
        return data_cloud

    # Try preferred path first, then fall back.
    if cloud_first:
        try:
            return run_cloud()
        except Exception as e:
            logger.warning(f"⚠️ Cloud (Groq) call failed for language '{language}', error: {e}")
            if cloud_only:
                logger.error("❌ Cloud-only mode active; not falling back to local Ollama.")
            else:
                try:
                    logger.info("🔁 Falling back to local Ollama (non-cloud-only mode).")
                    return run_local()
                except Exception as e2:
                    logger.error(f"❌ Local Ollama call failed after cloud fallback: {e2}")
    else:
        try:
            return run_local()
        except Exception as e:
            logger.warning(f"⚠️ Local Ollama call failed, falling back to cloud (Groq): {e}")
            try:
                return run_cloud()
            except Exception as e2:
                logger.error(f"❌ Groq fallback failed: {e2}")

    # --- FINAL FALLBACK: If both local and cloud fail ---
    logger.error("ℹ️ All AI models failed. Returning fallback.")
    cleaned_text, fallback_metrics = clean_stuttered_text(prompt)
    return {
        "text": cleaned_text,
        "english_translation": "",
        "metrics": fallback_metrics,
        "analysis": "<ul><li><strong>Processing Error</strong>: The AI analysis failed. Please check server logs. Applied basic heuristic cleanup.</li></ul>",
        "suggestions": "<ul><li>Try recording again with clearer speech.</li></ul>",
        "soap": {"s": "-", "o": "-", "a": "Processing Error", "p": "Retry"},
        "level": "Error",
        "classification": "Processing Failure",
        "demographics": "Unknown"
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
print(f"⏳ Loading Whisper '{whisper_model_size}' model with INT8 Quantization... (This may take a few minutes on first run)")

whisper_model = None
max_retries = 3
for attempt in range(max_retries):
    try:
        whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        break
    except Exception as e:
        logger.warning(f"⚠️ Failed to download/load Whisper model (Attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            print("⏳ Retrying in 5 seconds...")
            time.sleep(5)
        else:
            logger.critical("❌ Critical Error: Could not load Whisper model. Check your internet connection. Hugging Face might be blocked or temporarily unreachable.")
            sys.exit(1)

# --- Internationalization (i18n) ---
translations = {
    "en": {
        "title": "Stutter2Fluent",
        "header": "🎙️ Stutter2Fluent",
        "subheader": "Real-Time Speech-to-Speech AI Assistant",
        "start_button": "Start Recording",
        "stop_button": "Stop Recording",
        "status_ready": "Ready to connect...",
        "status_connected": "🔴 Connected & Listening...",
        "status_disconnected": "Disconnected",
        "status_stopped": "⏹️ Stopped",
        "transcription_placeholder": "Transcription will appear here...",
    },
}

@app.get("/")
async def get(request: Request, lang: str = "en"):
    lang_code = lang if lang in translations else "en"
    return templates.TemplateResponse("index.html", {"request": request, "translations": translations[lang_code]})

@app.get("/voices")
async def get_voices():
    """Endpoint to return available voices, preventing 404 errors in the UI."""
    # The frontend expects a dictionary mapping language codes to a list of voices.
    return {
        "en": [
            {"id": "af_heart", "name": "Heart (US English)"},
            {"id": "en-US-ChristopherNeural", "name": "Christopher (US English)"},
            {"id": "en-US-JennyNeural", "name": "Jenny (US English)"},
        ],
        "hi": [
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Hindi - Male)"},
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi - Female)"}
        ],
        "te": [
            {"id": "te-IN-MohanNeural", "name": "Mohan (Telugu - Male)"},
            {"id": "te-IN-ShrutiNeural", "name": "Shruti (Telugu - Female)"}
        ],
        "ta": [
            {"id": "ta-IN-ValluvarNeural", "name": "Valluvar (Tamil - Male)"},
            {"id": "ta-IN-PallaviNeural", "name": "Pallavi (Tamil - Female)"}
        ],
        "kn": [
            {"id": "kn-IN-GaganNeural", "name": "Gagan (Kannada - Male)"},
            {"id": "kn-IN-SapnaNeural", "name": "Sapna (Kannada - Female)"}
        ],
        "ml": [
            {"id": "ml-IN-MidhunNeural", "name": "Midhun (Malayalam - Male)"},
            {"id": "ml-IN-SobhanaNeural", "name": "Sobhana (Malayalam - Female)"}
        ]
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/api/process")
async def api_process(
    file: UploadFile = File(...),
    voice: Optional[str] = Form(None),
    tone: str = Form("Professional"),
    speed: float = Form(1.0),
    analysis_mode: str = Form("balanced"),
    language: str = Form("auto"),
    session_id: str = Form(None)
):
    # Check if this is a regeneration request.
    # We check content_type OR if the filename ends with .txt (fallback)
    is_text_input = file.content_type == 'text/plain' or file.filename.endswith('.txt')
    
    if is_text_input:
        logger.info("Processing text input from upload.")
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded text file is empty.")
        return await process_audio_pipeline(file_bytes, language, voice, speed, tone, session_id, analysis_mode, is_text_input=True, mime_type=file.content_type)
    
    # For audio, save to temp file to handle large sizes without memory limits
    # Ensure filename has extension (crucial for blobs/recordings)
    logger.info(f"Processing file upload: {file.filename} ({file.content_type})")
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
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id, analysis_mode, mime_type=file.content_type)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def get_voice_gender(voice_id: str) -> str:
    """Heuristic to determine gender from voice ID for fallback logic."""
    # Known female voices in the config
    female_names = ["Heart", "Sonia", "Neerja", "Swara", "Shruti", "Sapna", "Sobhana", "Pallavi"]
    if any(name in voice_id for name in female_names):
        return "Female"
    return "Male"

async def generate_openai_audio(text: str, voice_id: str) -> Optional[str]:
    """Generates audio using OpenAI TTS API as a fast cloud alternative."""
    if not OPENAI_API_KEY: return None
    
    # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
    valid_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    # Default to 'nova' (a great female voice) if the requested voice isn't an OpenAI voice
    openai_voice = voice_id if voice_id in valid_voices else "nova"
    
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1", # Extremely fast low-latency model
        "input": text,
        "voice": openai_voice
    }
    
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, json=data, headers=headers))
        if response.status_code == 200:
            filename = f"{uuid.uuid4()}.mp3"
            filepath = f"temp/{filename}"
            with open(filepath, "wb") as f:
                f.write(response.content)
            return f"/temp/{filename}"
        else:
            logger.error(f"❌ OpenAI TTS failed: {response.text}")
    except Exception as e:
        logger.error(f"❌ OpenAI TTS request failed: {e}")
    return None

async def tts_router(text: str, lang: str, voice_id: str = None, speed: float = SPEED) -> Optional[str]:
    """Selects a TTS engine based on language and returns a URL to the generated audio file."""
    logger.info(f"🎤 Routing TTS for language: {lang}, Voice: {voice_id}")

    if not text or not text.strip():
        print("⚠️ TTS received empty text. Skipping generation.")
        return None

    lang = lang.strip().lower()

    DEFAULT_VOICES = {
        "en": "af_heart",
        "hi": "hi-IN-MadhurNeural",
        "te": "te-IN-MohanNeural",
        "ta": "ta-IN-ValluvarNeural",
        "kn": "kn-IN-GaganNeural",
        "ml": "ml-IN-MidhunNeural",
    }

    # Validate voice_id: if a voice is provided, it must be for the correct language.
    # A voice is valid if its language code (e.g., 'en-' from 'en-US-JennyNeural') is in the target language.
    is_valid_voice = False
    if voice_id:
        # Kokoro voices are special (af_...) and only for English
        if voice_id.startswith("af_"):
            if lang == "en":
                is_valid_voice = True
        # For EdgeTTS voices (e.g., 'en-US-JennyNeural'), check if the language prefix matches.
        elif "-" in voice_id:
            voice_lang_prefix = voice_id.split('-')[0].lower()
            if voice_lang_prefix == lang:
                is_valid_voice = True

    if not is_valid_voice:
        logger.warning(f"⚠️ Invalid or mismatched voice '{voice_id}' for language '{lang}'. Falling back to default.")
        # Fallback to the default voice for the target language.
        voice_id = DEFAULT_VOICES.get(lang, "en-US-ChristopherNeural") # Final fallback to a known good voice

    # Check if we should use Kokoro (English only, model loaded, and voice is a Kokoro voice)
    use_kokoro = lang == "en" and kokoro is not None and voice_id.startswith("af_")

    # Route cloud requests to OpenAI TTS instead of ElevenLabs
    prefer_cloud_tts = False # Disabled to prevent OpenAI quota errors and force free Edge-TTS
    
    if prefer_cloud_tts:
        logger.info(f"   ☁️ Using OpenAI TTS (Cloud) for '{lang}'...")
        audio_url = await generate_openai_audio(text, voice_id)
        if audio_url:
            return audio_url
        logger.warning("⚠️ OpenAI TTS failed. Falling back to local Edge-TTS/Kokoro...")

    audio_data = b""
    mime_type = "audio/wav"

    if use_kokoro:
        logger.info(f"   🚀 Using Kokoro TTS (Local) - {voice_id}")
        # Run blocking Kokoro call in executor
        loop = asyncio.get_running_loop()
        samples, sample_rate = await loop.run_in_executor(None, lambda: kokoro.create(text, voice=voice_id, speed=speed, lang="en-us"))

        # Convert Float32 to Int16 PCM for browser compatibility
        # Many browsers/players play float32 WAV as silence
        if isinstance(samples, np.ndarray) and samples.dtype == np.float32:
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767).astype(np.int16)

        # Write to in-memory buffer
        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, samples)
        audio_data = buffer.getvalue()
    else:
        # Fallback: If voice_id is a Kokoro voice but we can't use Kokoro, switch to Edge-TTS default
        if voice_id.startswith("af_"):
            voice_id = "en-US-ChristopherNeural"

        # Use edge-tts for other languages
        logger.info(f"   🚀 Using edge-tts (Local) - {voice_id}")
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
            fallback_voice = "en-US-JennyNeural" if "Christopher" in voice_id else "en-US-ChristopherNeural"
            ALTERNATE_VOICES = {
                "en": "en-US-JennyNeural",
                "hi": "hi-IN-SwaraNeural",
                "te": "te-IN-ShrutiNeural",
                "ta": "ta-IN-PallaviNeural",
                "kn": "kn-IN-SapnaNeural",
                "ml": "ml-IN-SobhanaNeural"
            }

            # Smart Fallback: Try default voice for the language if the requested one fails
            fallback_voice = DEFAULT_VOICES.get(lang, "en-US-ChristopherNeural")
            if fallback_voice == voice_id:
                fallback_voice = ALTERNATE_VOICES.get(lang, "en-US-JennyNeural")
            if fallback_voice:
                logger.info(f"🔄 Retrying with fallback voice: {fallback_voice}")
                try:
                    communicate = edge_tts.Communicate(text, fallback_voice, rate=rate_str)
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    mime_type = "audio/mpeg"
                except Exception as e2:
                    logger.error(f"❌ Fallback EdgeTTS failed: {e2}") # audio_data will be empty
            else:
                pass # audio_data will be empty
        except Exception as e:
            logger.error(f"❌ EdgeTTS failed: {e}") # audio_data will be empty

    if audio_data:
        ext = ".mp3" if mime_type == "audio/mpeg" else ".wav"
        filename = f"{uuid.uuid4()}{ext}"
        filepath = f"temp/{filename}"
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return f"/temp/{filename}"

    # --- CLOUD FALLBACK: OpenAI ---
    if False: # Disabled OpenAI fallback
        logger.warning("⚠️ Local TTS failed. Falling back to OpenAI (Cloud)...")
        audio_url = await generate_openai_audio(text, voice_id)
        if audio_url:
            return audio_url

    logger.error("❌ All TTS engines failed.")
    return None

@app.post("/process/upload")
async def process_upload(
    file: UploadFile = File(...), 
    voice: Optional[str] = Form(None),
    language: str = Form("auto"),
    speed: float = Form(None),
    analysis_mode: str = Form("balanced"),
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
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id, analysis_mode, is_text_input=is_text, mime_type=file.content_type)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/process/url")
async def process_url(
    url: str = Form(...), 
    voice: Optional[str] = Form(None),
    language: str = Form("auto"),
    speed: float = Form(None),
    analysis_mode: str = Form("balanced"),
    tone: str = Form("Professional"),
    session_id: str = Form(None)
):
    logger.info(f"Processing URL: {url}")
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
        mime_type = mimetypes.guess_type(temp_filename)[0] or "audio/wav"
        return await process_audio_pipeline(temp_filename, language, voice, speed, tone, session_id, analysis_mode, mime_type=mime_type)
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
    voice: Optional[str] = Form(None)
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


SUPPORTED_EXTENSIONS = {
    # Audio
    '.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg', '.opus', '.wma',
    # Video
    '.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v', '.3gp'
}

def transcribe_with_deepgram(audio_data: Union[bytes, str], language: str = None, mimetype: str = "audio/wav") -> Optional[dict]:
    """Transcribes audio using Deepgram Nova-3/Nova-2, with language-aware model selection."""
    if not DEEPGRAM_API_KEY: return None

    # Ensure mimetype is valid (Deepgram requires specific content-types)
    if not mimetype or mimetype == "application/octet-stream":
        mimetype = "audio/wav"

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mimetype
    }

    # Prepare data once
    try:
        if isinstance(audio_data, str):
            with open(audio_data, "rb") as f:
                data = f.read()
        else:
            data = audio_data
    except Exception as e:
        logger.error(f"❌ Failed to read audio data for Deepgram: {e}")
        return None

    # Models to try in order: nova-3 (newest), nova-2 (standard), general (fallback)
    models = ["nova-3", "nova-2", "general"]
    # Language-aware model selection: South Indian languages + complex scripts only work with nova-3
    # For other languages/auto-detect, try nova-3 → nova-2 → general
    if language and language in SOUTH_INDIAN_LANGS:
        models = ["nova-3"]  # South Indian scripts only support nova-3
        logger.info(f"ℹ️  Using nova-3 only for South Indian language: {language}")
    elif language and language in COMPLEX_LANGUAGES:
        models = ["nova-3"]  # Complex scripts (like Hindi) prefer nova-3
        logger.info(f"ℹ️  Using nova-3 only for complex language: {language}")
    else:
        models = ["nova-3", "nova-2", "general"]  # English and auto-detect: try all

    for model in models:
        params = {
            "model": model,
            "smart_format": "true",
            "filler_words": "true",
            "punctuate": "true",
        }
        if language and language != "auto":
            params["language"] = language
        else:
            params["detect_language"] = "true"

        url = f"https://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"

        response = None
        try:
            response = requests.post(url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Safely access the transcript
            try:
                transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
                if transcript: # Ensure transcript is not empty
                    detected_lang = result['results']['channels'][0].get('detected_language', 'en')
                    return {"text": transcript, "language": detected_lang}
                else:
                    logger.warning(f"⚠️ Deepgram returned an empty transcript for model '{model}'.")
            except (KeyError, IndexError):
                logger.warning(f"⚠️ Deepgram response for model '{model}' was malformed and did not contain a transcript.")
                # Continue to next model
        except requests.exceptions.HTTPError as e:
            # Check if it's a model availability issue (400 Bad Request)
            if response is not None and response.status_code == 400 and model != models[-1]:
                logger.warning(f"⚠️ Deepgram model '{model}' failed (likely unsupported language). Retrying with fallback model...")
                continue

            # If it's another error or we ran out of models
            details = ""
            if response is not None:
                try:
                    details = f" | Deepgram Response: {response.text}"
                except: pass
            logger.error(f"❌ Deepgram STT failed: {e}{details}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Deepgram STT request failed: {e}")
            return None # Exit on connection error, no need to retry other models
        except json.JSONDecodeError as e:
            details = ""
            if response is not None:
                details = f" | Response: {response.text[:200]}"
            logger.error(f"❌ Failed to decode Deepgram JSON response (model: {model}): {e}{details}")

    return None

async def process_audio_pipeline(input_data: Union[bytes, str], lang_pref, voice_pref, speed_pref=None, tone_pref="Professional", session_id=None, analysis_mode="balanced", is_text_input=False, mime_type: str = "audio/wav"):
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
            logger.info("Pipeline started for TEXT input.")
            # 1. Bypass Transcription for Text Input (Regeneration)
            if isinstance(input_data, bytes):
                transcribed_text = input_data.decode('utf-8')
            else:
                with open(input_data, "r", encoding="utf-8") as f:
                    transcribed_text = f.read()
            detected_lang = lang_pref if lang_pref != "auto" else "en" # Default or trust pref
            
            if lang_pref == "auto":
                if any('\u0900' <= c <= '\u097F' for c in transcribed_text): detected_lang = "hi"
                elif any('\u0C00' <= c <= '\u0C7F' for c in transcribed_text): detected_lang = "te"
                elif any('\u0B80' <= c <= '\u0BFF' for c in transcribed_text): detected_lang = "ta"
                elif any('\u0C80' <= c <= '\u0CFF' for c in transcribed_text): detected_lang = "kn"
                elif any('\u0D00' <= c <= '\u0D7F' for c in transcribed_text): detected_lang = "ml"
                else: detected_lang = "en"
            else:
                detected_lang = lang_pref
        else:
            logger.info("Pipeline started for AUDIO/VIDEO input.")
            # 1. Transcribe Audio & Extract Features (Parallel)
            async def get_acoustic_features():
                try:
                    audio_source_for_decode = input_data
                    if isinstance(input_data, bytes):
                        audio_source_for_decode = io.BytesIO(input_data)

                    # decode_audio handles file paths (str) and file-like objects (BytesIO)
                    # Run in executor to avoid blocking main thread
                    audio_np = await loop.run_in_executor(None, lambda: decode_audio(audio_source_for_decode, sampling_rate=16000))

                    # Extract Acoustic Features (RMS, ZCR)
                    features = extract_acoustic_features(audio_np)
                    print(f"📊 Acoustic Features Extracted: RMS={features.get('rms',0):.3f}, ZCR={features.get('zcr',0):.3f}, F0={features.get('f0',0):.1f}Hz")
                    del audio_np
                    return features
                except Exception as e:
                    logger.warning(f"⚠️ Acoustic Feature Extraction failed: {e}")
                    return {"rms": 0.0, "zcr": 0.0, "variance": 0.0, "f0": 0.0}

            # Execute parallel
            feature_task = asyncio.create_task(get_acoustic_features())

            transcribed_text = ""
            detected_lang = "en"
            segments = []
            info = None

            # --- STT STRATEGY: Local Whisper First, Deepgram only as fallback ---
            logger.info("🎤 Preferring local Whisper (offline) for transcription, with Deepgram as a fallback if needed.")

            tiers_to_try = [ACTIVE_TIER]
            if ACTIVE_TIER == "low_latency":
                tiers_to_try.append("balanced")
            elif ACTIVE_TIER == "balanced":
                tiers_to_try.append("high_quality") # Fallback: Try beam search if greedy fails

            for attempt_tier in tiers_to_try: 
                def transcribe_blocking(tier):
                    tier_settings = TIER_CONFIG.get(tier, TIER_CONFIG["low_latency"])

                    # --- Dynamic Transcription Tuning ---
                    # Non-English languages, especially with stuttering, benefit from a wider beam search
                    # and a more sensitive speech detection threshold.
                    current_no_speech_threshold = 0.6
                    current_beam_size = tier_settings.get("beam_size", 5)

                    # --- High-Resolution Multilingual Logic ---
                    transcribe_kwargs = {
                        "beam_size": current_beam_size,
                        "best_of": 5,
                        "patience": 2.0, # Allows deeper exploration of the audio signal
                        "repetition_penalty": 1.0,
                        "no_speech_threshold": current_no_speech_threshold,
                        "word_timestamps": True,
                        "vad_filter": True, # Enabled VAD to eliminate disturbing background noises
                        "vad_parameters": dict(min_silence_duration_ms=2000, speech_pad_ms=400), # 2s tolerance preserves stutter blocks without cutting off
                        "condition_on_previous_text": False,
                        "log_prob_threshold": None, # Force transcription even for low-confidence audio (essential for stuttering)
                    }
                    logger.info(f"🎤 Transcribing with Whisper ({whisper_model_size} model, tier={tier}, beam_size={current_beam_size})...")
                    try:
                        # Fallback to input_data since audio_np is local to feature extraction task
                        source = io.BytesIO(input_data) if isinstance(input_data, bytes) else input_data

                        # Dynamic language setting for Whisper
                        transcribe_kwargs["language"] = lang_pref if lang_pref != "auto" else None
                        # Use priming prompt only if available for the specified language

                        if lang_pref in PRIMING_PROMPTS:
                            transcribe_kwargs["initial_prompt"] = PRIMING_PROMPTS[lang_pref]
                        transcribe_kwargs["temperature"] = (0.0, 0.2, 0.4) # Keep temperature for robustness

                        segments_gen, info = whisper_model.transcribe(source, **transcribe_kwargs)
                        return list(segments_gen), info
                    except Exception as e:
                        logger.error(f"❌ Whisper Transcription Error: {e}")
                        return [], None
                
                segments, info = await loop.run_in_executor(None, lambda: transcribe_blocking(attempt_tier))

                if info is None:
                    logger.warning(f"⚠️ Tier {attempt_tier} failed (info is None). Retrying with next tier...")
                    continue

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
                        continue

            acoustic_features = await feature_task

            # --- CLOUD FALLBACK STT (if local Whisper fails) ---
            if not transcribed_text and DEEPGRAM_API_KEY:
                logger.info("⚠️ Local Whisper failed to produce a transcript. Falling back to Deepgram (Cloud)...")
                dg_result = await loop.run_in_executor(None, lambda: transcribe_with_deepgram(input_data, lang_pref, mime_type))
                if dg_result and dg_result.get("text"):
                    transcribed_text = dg_result["text"]
                    detected_lang = dg_result.get("language", "en")
                    # Create a dummy `info` object so the pipeline doesn't fail later
                    class DummyInfo: duration = 30.0
                    info = DummyInfo()

            if not transcribed_text:
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
            
            effective_duration = speech_duration if speech_duration > 0.5 else (info.duration if info else 10.0)
            input_wpm = (word_count / effective_duration) * 60 if effective_duration > 0 else 150
        
        if speed_pref:
            final_speed = float(speed_pref)
        else:
            final_speed = max(0.8, min(1.3, input_wpm / 150)) # Clamp speed to natural range

        # Memory Cleanup: Segments no longer needed
        if 'segments' in locals(): del segments
        gc.collect()

        # --- Improved Language Auto-Detection Logic ---
        final_lang = detected_lang if lang_pref == "auto" else lang_pref
        logger.info(f"🔍 Language set to: '{final_lang}'")

        # Map code to full name for Agent (e.g., "te" -> "Telugu") for better LLM accuracy
        agent_lang_name = LANGUAGE_NAMES.get(final_lang, final_lang)

        # Normalize analysis mode (front-end uses 'speed'/'deep' labels)
        analysis_mode_normalized = str(analysis_mode).lower() if analysis_mode else "balanced"
        if analysis_mode_normalized in ("speed", "fast", "low_latency"):
            analysis_mode = "low_latency"
        elif analysis_mode_normalized in ("deep", "high", "high_quality"):
            analysis_mode = "high_quality"
        else:
            analysis_mode = "balanced"

        # 2. AI Analysis
        # Run blocking AI call in a separate thread to prevent server freeze
        logger.info(f"Running AI analysis for language '{agent_lang_name}' (mode={analysis_mode})...")

        # Increased timeouts to give local models more time, especially on CPU.
        analysis_timeouts = {
            "low_latency": 25,   # ~25 seconds for fastest model
            "balanced": 120,     # 120s for llama3.1:3b on CPU (increased for timeout fix)
            "high_quality": 120  # 2 minutes for the largest model
        }
        timeout_secs = analysis_timeouts.get(analysis_mode, 20)

        try:
            ai_data = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: knowledge_agent_client(
                    transcribed_text,
                    language=agent_lang_name,
                    tone=tone_pref,
                    session_id=session_id,
                    acoustic_features=acoustic_features,
                    analysis_mode=analysis_mode,
                    prefer_cloud=should_prefer_cloud(agent_lang_name)
                )),

                timeout=timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ AI analysis timed out after {timeout_secs}s (mode={analysis_mode}). Attempting cloud fallback if available.")
            ai_data = None

            # Fallback to cloud if the local Ollama path is too slow or unavailable.
            if GROQ_API_KEY:
                fallback_timeout = max(timeout_secs * 2, 20)
                logger.info(f"⏭️ Trying cloud fallback (Groq) with {fallback_timeout}s timeout...")
                try:
                    ai_data = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: knowledge_agent_client(
                            transcribed_text,
                            language=agent_lang_name,
                            tone=tone_pref,
                            session_id=session_id,
                            acoustic_features=acoustic_features,
                            analysis_mode=analysis_mode,
                            prefer_cloud=True
                        )),
                        timeout=fallback_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Cloud fallback also timed out after {fallback_timeout}s.")
                except Exception as e:
                    logger.error(f"❌ Cloud fallback (Groq) failed: {e}")

            if not ai_data:
                ai_data = {
                    "text": transcribed_text,
                    "english_translation": "N/A",
                    "analysis": "<ul><li><strong>Timeout</strong>: AI response took too long. Try switching to 'low_latency' mode, ensuring Ollama is running locally, or providing a Groq API key for fallback.</li></ul>",
                    "suggestions": "<ul><li>Try again or use a faster mode (low_latency).</li></ul>",
                    "metrics": {"words": len(transcribed_text.split()) if transcribed_text else 0, "disfluencies": 0, "rate": 100},
                    "soap": {"s": "N/A", "o": "N/A", "a": "Processing Timeout", "p": "Retry"},
                    "level": "Error",
                    "classification": "Timeout",
                    "demographics": "Unknown"
                }

        if not ai_data:
            cleaned_text, fallback_metrics = clean_stuttered_text(transcribed_text)
            # Unified Fallback for connection error or timeout
            # Fallback for connection error
            ai_data = {
                "text": cleaned_text,
                "english_translation": "N/A",
                "analysis": "<ul><li><strong>Processing Error / Timeout</strong>: The AI response could not be parsed or took too long.</li><li>Applied basic heuristic cleanup to remove stutters.</li></ul>",
                "suggestions": "<ul><li>Try recording again with clearer speech.</li><li>Consider using a faster mode or checking API keys.</li></ul>",
                "metrics": fallback_metrics,
                "soap": {"s": "N/A", "o": "N/A", "a": "Processing Error", "p": "Retry"},
                "level": "Error",
                "classification": "Parsing/Timeout Failure",
                "demographics": "Unknown"
            }

        # 3. TTS
        corrected_text = ai_data.get("text", "")
        logger.info(f"Synthesizing speech for corrected text in language '{final_lang}'...")
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
            "response": {**ai_data, "audio_url": output_audio_url, "input_audio_url": input_audio_url, "input_text": transcribed_text, "acoustic_features": acoustic_features}, # For new UI compatibility
            "metrics": {"wer": wer_score},
            "detected_language": detected_lang,
            "status": "success",
            "transcription": transcribed_text
        }
    except Exception as e:
        logger.error("❌ Pipeline Error:")
        traceback.print_exc()
        # Return proper error response with analysis structure so frontend can display it
        return {
            "status": "error",
            "message": str(e),
            "input_audio_url": input_audio_url,
            "transcription": "(Error)",
            "response": {
                "text": "(Error)",
                "analysis": f"<ul><li><strong>Transcription Failed</strong>: {str(e)}</li><li>Please ensure your audio file contains clear speech and is not silent/corrupted.</li><li>Check your microphone levels and try again.</li></ul>",
                "suggestions": "<ul><li>📍 <strong>Recording Tips:</strong></li><li>Speak clearly and loudly</li><li>Minimize background noise</li><li>Use a good quality microphone</li><li>Ensure the audio file is not corrupted</li></ul>",
                "soap": {
                    "s": "Audio transcription failed - unable to process input",
                    "o": "No speech detected or audio file is corrupted",
                    "a": "Transcription Error - technical issue with input processing",
                    "p": "Retry with a valid audio recording containing clear speech"
                },
                "metrics": {"words": 0, "disfluencies": 0, "rate": 0},
                "level": "Error",
                "classification": "Input Error",
                "demographics": "Unknown"
            }
        }


def main():
    app_host = os.getenv("HOST", "127.0.0.1")
    app_port = int(os.getenv("PORT", 9067))
    print(f"🚀 Starting Stutter2Fluent ({ACTIVE_TIER} tier)...")
    print(f"ℹ️  Voice Profile: {VOICE_PROFILE}, Speed: {SPEED}x")
    print(f"👉 Open your browser at http://{app_host}:{app_port}")
    print(f"   (Set the 'PORT' environment variable to use a different port if {app_port} is busy)")
    # webbrowser.open(f"http://localhost:{app_port}") # Disable auto-open for Docker/Server envs
    uvicorn.run(app, host=app_host, port=app_port)

if __name__ == "__main__":
    main()
