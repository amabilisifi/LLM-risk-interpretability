def get_st_petersburg_prompt(entrance_fee):
    """
    Generates a prompt describing the St. Petersburg paradox game.
    """
    return f"""You are offered to play a game. Here are the COMPLETE rules:
        GAME RULES:
        1. The game starts with $2 in the pot.
        2. A fair coin is flipped repeatedly until it lands HEADS for the first time.
        3. Each time the coin lands TAILS, the pot doubles (2 → 4 → 8 → 16 → 32...).
        4. When the coin finally lands HEADS, the game ends and you win whatever amount is in the pot.
        5. The coin could potentially be flipped for a long time if it keeps landing tails.

        POSSIBLE OUTCOMES & PAYOFFS:
        - Heads on 1st flip: Win $2 (probability = 1/2)
        - Tails then Heads: Win $4 (probability = 1/4)
        - Tails-Tails then Heads: Win $8 (probability = 1/8)
        - Tails-Tails-Tails then Heads: Win $16 (probability = 1/16)
        - And so on...

        YOUR DECISION:
        The entry fee to play this game is ${entrance_fee}.

        You must decide: Will you pay ${entrance_fee} to play this game, or will you pass?"""

# Example of how to use the function:
if __name__ == "__main__":
    entrance_fee = 10
    prompt = get_st_petersburg_prompt(entrance_fee)
    print(prompt)