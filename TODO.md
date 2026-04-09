# Fluency-Net Issue Fixes - Detailed Implementation Steps

**Status: Approved by user. Implementing step-by-step.**

## 1. ✅ Create this TODO.md (Current)

## 2. [PENDING] Add cleaning functions to main.py

- `enforce_fluent_output(text, orig_text, language)`: Primary, language-aware regex cleaning.
- `clean_stuttered_text_by_language(text, language)`: Secondary.
- `clean_stuttered_text(text)`: Fallback generic.
- Goal: 90%+ pass on eval_pipeline.py (15 multilingual samples).
- Test: `python eval_pipeline.py`

## 3. [PENDING] Implement LOCAL_ONLY_EN logic in knowledge_agent_client()

- Default: `LOCAL_ONLY_EN=true`
- English: local Ollama only → heuristic fallback on timeout (no Groq).
- Non-English: Groq cloud-first + Deepgram STT.

## 4. [PENDING] Update get_ai_agent() model selection

- low_latency/speed: llama3.2:1b
- balanced/high_quality: llama3.1:8b

## 5. [PENDING] Prefer Deepgram STT for non-en if API key set

- Check DEEPGRAM_API_KEY → use first.

## 6. [INSTALL - User]

```
ollama pull llama3.1:8b llama3.2:1b
pip install -r requirements.txt
python download_requirements.py  # Fewer warnings expected
```

## 7. [TEST - User]

```
python eval_pipeline.py  # Expect 90%+ accuracy
python main.py           # Visit http://localhost:9067
```

## 8. [COMPLETION] Update this TODO.md with results, attempt_completion

**Next Action: Edit main.py with cleaning functions + env logic.**
