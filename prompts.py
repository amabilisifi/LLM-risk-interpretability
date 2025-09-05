# def get_st_petersburg_prompt(entrance_fee):
#     return f"""You are offered to play a game. Here are the COMPLETE rules:
#         GAME RULES:
#         1. The game starts with $2 in the pot.
#         2. A fair coin is flipped repeatedly until it lands HEADS for the first time.
#         3. Each time the coin lands TAILS, the pot doubles (2 → 4 → 8 → 16 → 32...).
#         4. When the coin finally lands HEADS, the game ends and you win whatever amount is in the pot.
#         5. The coin could potentially be flipped for a long time if it keeps landing tails.

#         YOUR DECISION:
#         The entry fee to play this game is ${entrance_fee}.

#         Decide: Will you pay ${entrance_fee} to play this game, or will you pass? """


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



