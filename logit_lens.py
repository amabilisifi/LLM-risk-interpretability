import torch
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def apply_logit_lens(model, tokenizer, output_path):
    # Load saved outputs
    try:
        data = torch.load(output_path)
        hidden_states = data["hidden_states"][-1]  # Last generation step (tuple of per-layer states)
        sequences = data["sequences"]              # Token IDs
    except Exception as e:
        logger.error(f"Failed to load saved outputs from {output_path}: {e}")
        return {
            "num_layers": 0,
            "sequence_length": 0,
            "last_token": "",
            "layers": [],
            "note": "Failed to load saved outputs for logit lens analysis."
        }

    logger.info("Num layers: %d", len(hidden_states))
    logger.info("Sequence length: %d", sequences.shape[1])

    # Get the last token ID to decode alongside
    last_token_id = sequences[0, -1].item()
    last_token_text = tokenizer.decode([last_token_id])

    logger.info("True last token: '%s'", last_token_text)

    # Get token IDs for "play"/"pass" and capitalized variants
    play_lower_ids = tokenizer.encode("play", add_special_tokens=False)
    pass_lower_ids = tokenizer.encode("pass", add_special_tokens=False)
    play_cap_ids = tokenizer.encode("Play", add_special_tokens=False)
    pass_cap_ids = tokenizer.encode("Pass", add_special_tokens=False)
    play_token_id = play_cap_ids[0] if play_cap_ids else play_lower_ids[0]
    pass_token_id = pass_cap_ids[0] if pass_cap_ids else pass_lower_ids[0]

    # Collect logit lens results
    logit_lens_results = {
        "num_layers": len(hidden_states),
        "sequence_length": sequences.shape[1],
        "last_token": last_token_text,
        "note": "Play/Pass probabilities may be near 0 if the last token is a JSON syntax token (e.g., '}'). Consider analyzing earlier tokens for decision-related probabilities.",
        "layers": []
    }

    for layer_idx, h in enumerate(hidden_states):
        # Take hidden state for last token
        state = h[0, -1, :]   # (hidden_dim,)

        with torch.no_grad():
            logits = model.lm_head(state)   # (vocab_size,)
            probs = torch.softmax(logits, dim=-1)

        topk = torch.topk(probs, 5)
        decoded = [tokenizer.decode([i.item()]) for i in topk.indices]
        top_probs = [float(v) for v in topk.values]

        # Extract play/pass probabilities
        play_prob = float(probs[play_token_id]) if play_token_id is not None else 0.0
        pass_prob = float(probs[pass_token_id]) if pass_token_id is not None else 0.0

        logger.info("Layer %02d: %s  (top prob = %.3f)", layer_idx, decoded, top_probs[0])
        logger.info("  Play probability: %.3f, Pass probability: %.3f", play_prob, pass_prob)

        # Store results for this layer
        logit_lens_results["layers"].append({
            "layer_index": layer_idx,
            "top_tokens": decoded,
            "top_probabilities": top_probs,
            "play_probability": play_prob,
            "pass_probability": pass_prob
        })
        
    return logit_lens_results

def analyze_logit_lense(logit_lens_results, timestamp):
    if not logit_lens_results["layers"]:
        logger.warning("No layers to analyze.")
        return None

    results = []
    layers = []
    play_probs = []
    pass_probs = []
    risk_scores = []

    for layer in logit_lens_results["layers"]:
        play_p = layer["play_probability"]
        pass_p = layer["pass_probability"]
        risk = play_p - pass_p
        results.append({
            "layer": layer["layer_index"],
            "play_probability": float(play_p),
            "pass_probability": float(pass_p),
            "risk_score": float(risk),
            "top_tokens": layer["top_tokens"][:5],
        })
        layers.append(layer["layer_index"])
        play_probs.append(play_p)
        pass_probs.append(pass_p)
        risk_scores.append(risk)

    # Add insights
    max_risk_layer = layers[risk_scores.index(max(risk_scores))]
    min_risk_layer = layers[risk_scores.index(min(risk_scores))]
    trend_summary = "Play prob increases over layers" if play_probs[-1] > play_probs[0] else "Pass prob dominates later layers"
    insights = {
        "max_risk_layer": max_risk_layer,
        "min_risk_layer": min_risk_layer,
        "trend_summary": trend_summary,
        "average_play_prob": sum(play_probs) / len(play_probs),
        "average_pass_prob": sum(pass_probs) / len(pass_probs),
    }
    full_results = {"results": results, "insights": insights}

    # Save enhanced JSON
    json_path = f"outputs/logit_lens_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Logit lens JSON saved to {json_path}")

    # Plot probabilities and risk score
    plt.figure(figsize=(10, 6))
    plt.plot(layers, play_probs, label='Play Probability', marker='o')
    plt.plot(layers, pass_probs, label='Pass Probability', marker='o')
    plt.plot(layers, risk_scores, label='Risk Score (Play - Pass)', marker='x', linestyle='--')
    plt.xlabel('Layer Index')
    plt.ylabel('Probability / Score')
    plt.title('Logit Lens Analysis: Play vs Pass Over Layers')
    plt.legend()
    plt.grid(True)
    plot_path = f"outputs/logit_lens_plot_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Logit lens plot saved to {plot_path}")

    # Return for printing
    return full_results