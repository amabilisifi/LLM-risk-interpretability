import json
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def _clean_json_like(s: str) -> str:
    # Remove common noise: backticks, surrounding markdown fences
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s)
    s = re.sub(r"```$", "", s)
    # Remove leading/trailing text markers
    s = s.strip()
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*(\}|])", r"\1", s)
    # Try to replace single quotes if there are no double quotes (naive)
    if s.count('"') < s.count("'"):
        s = s.replace("'", '"')
    return s

def extract_first_json_dict(text: str) -> Optional[dict]:
    """
    Find the first balanced {...} block in text and attempt to parse it as JSON.
    Returns dict or None.
    """
    logger.debug("Attempting to extract JSON from text:\n%s", text)
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_indices:
        stack = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                stack += 1
            elif text[i] == "}":
                stack -= 1
                if stack == 0:
                    candidate = text[start:i+1]
                    # Try raw parse
                    try:
                        parsed = json.loads(candidate)
                        logger.debug("Successfully parsed JSON:\n%s", parsed)
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.debug("Raw JSON parse failed: %s", e)
                        # Try cleaning and parse
                        try:
                            cleaned = _clean_json_like(candidate)
                            logger.debug("Cleaned JSON candidate:\n%s", cleaned)
                            parsed = json.loads(cleaned)
                            logger.debug("Successfully parsed cleaned JSON:\n%s", parsed)
                            return parsed
                        except Exception as e:
                            logger.debug("Cleaned JSON parse failed: %s", e)
                            break  # Give up on this start, try next
    logger.warning("No valid JSON found in text")
    return None

def _clamp(x, low, high):
    try:
        v = float(x)
    except Exception:
        return None
    return max(low, min(high, v))

def sanitize_reasoning(r: str) -> str:
    """Remove obvious prompt leakage like the rules or 'GAME RULES' sections."""
    if not isinstance(r, str):
        return ""
    # Truncate if appears to contain the rules copy
    markers = ["GAME RULES", "YOUR TASK", "OUTPUT FORMAT", "SAMPLE_INPUT", "SAMPLE_OUTPUT"]
    for m in markers:
        idx = r.find(m)
        if idx != -1:
            r = r[:idx].strip()
    # Small cleanup
    r = r.strip()
    # Remove repeated blocks (very long repeats)
    if len(r) > 700:
        r = r[:600].rstrip() + "..."
    return r

def generate_json_output(model_response: str, entrance_fee: float):
    """
    Robust pipeline:
    1) Try to extract first JSON object from model_response.
    2) Validate and normalize to expected schema.
    3) Fallback to a safe default if nothing found.
    """
    parsed = extract_first_json_dict(model_response)
    if parsed is None:
        # No JSON found — produce a helpful fallback
        logger.warning("Falling back to default JSON output")
        return {
            "play_or_pass": "Pass",
            "reasoning": "Failed to parse JSON output from model.",
            "probability": 0.5,
            "confidence": 0.5,
            "log_probability": -1.0,
            "entrance_fee": entrance_fee,
        }

    # Normalize field names (accept synonyms)
    out = {}
    # Some models use "decision" instead of "play_or_pass"
    out['play_or_pass'] = parsed.get("play_or_pass") or parsed.get("decision") or parsed.get("choice") or "Pass"
    if isinstance(out['play_or_pass'], str):
        s = out['play_or_pass'].strip().lower()
        out['play_or_pass'] = "Play" if s.startswith("p") and not s.startswith("pa") else ("Play" if s == "play" else ("Pass" if s == "pass" else ("Play" if "play" in s else "Pass")))
        # Fallback: normalize more simply
        if out['play_or_pass'] not in ("Play", "Pass"):
            out['play_or_pass'] = "Play" if "play" in s else "Pass"

    # Reasoning: sanitize from prompt leakage
    out['reasoning'] = sanitize_reasoning(parsed.get("reasoning", ""))

    # Numeric fields (coerce and clamp)
    out['probability'] = _clamp(parsed.get("probability", parsed.get("prob", 0.5)), 0.0, 1.0) or 0.5
    out['confidence'] = _clamp(parsed.get("confidence", parsed.get("conf", 0.5)), 0.0, 1.0) or 0.5
    # log_probability can be any float; fallback to -1.0
    try:
        out['log_probability'] = float(parsed.get("log_probability", parsed.get("logprob", -1.0)))
    except Exception:
        out['log_probability'] = -1.0

    out['entrance_fee'] = entrance_fee

    logger.info("Generated JSON output:\n%s", out)
    return out