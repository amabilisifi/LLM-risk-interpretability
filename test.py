import torch

data = torch.load("./outputs/weights_attn_GEMMA_2B_20250904_231745.pt")

print(data.keys())

# sequences
print("Sequences shape:", data["sequences"].shape)

# hidden states: tuple of steps
print("Num generation steps:", len(data["hidden_states"]))

# take the last generation step
last_step = data["hidden_states"][-1]   # tuple of layer states
print("Num layers:", len(last_step))
print("Shape of layer 0 hidden state:", last_step[0].shape)  
# should be (batch, seq_len, hidden_dim)
