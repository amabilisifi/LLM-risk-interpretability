
def get_st_petersburg_prompt(entrance_fee: int) -> str:
    return f"""INSTRUCTIONS:
Return exactly one JSON object with these fields:
- "play_or_pass": string ("Play" or "Pass"),
- "reasoning": 1–3 sentences giving a clear justification (cannot be empty, cannot repeat the rules/instructions),
- "probability": number (0.0–1.0),
- "confidence": number (0.0–1.0),
- "log_probability": number,
- "entrance_fee": number

TASK:
You are evaluating the St. Petersburg game:
- Pot starts at $2, coin flips until first HEADS, doubling each TAIL.
- Entry fee: ${entrance_fee}.

Respond ONLY with the JSON object.
"""

def get_st_petersburg_prompt_cot(entrance_fee: int) -> str:
    return f"""INSTRUCTIONS:
Return exactly one JSON object with these fields:
- "play_or_pass": string ("Play" or "Pass"),
- "reasoning": 1–3 sentences giving a clear justification (cannot be empty, cannot repeat the rules/instructions),
- "probability": number (0.0–1.0),
- "confidence": number (0.0–1.0),
- "log_probability": number,
- "entrance_fee": number

TASK:
You are evaluating the St. Petersburg game:
- Pot starts at $2, coin flips until first HEADS, doubling each TAIL.
- Entry fee: ${entrance_fee}.

Use the following Chain of Thought process to determine your response:
1. Calculate the expected value (EV) of the game by considering the infinite series of possible outcomes (e.g., $2 for 1 flip, $4 for 2 flips, $8 for 3 flips, etc., with probabilities 1/2, 1/4, 1/8, ...).
2. Compare the EV to the entrance fee to assess if the game is worth playing based on raw financial gain.
3. Consider practical risk factors, such as the potential for infinite losses or diminishing returns, and how they might affect a rational decision.
4. Decide on "Play" or "Pass" based on the balance of EV, fee, and risk, then assign probability and confidence reflecting this reasoning.

Respond ONLY with the JSON object.
"""

