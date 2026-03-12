from src.data import *

def generate_answer(model, example: str, vocab: Vocab, mode: str,max_new_tokens: int = 10):
    model.eval()
    device = next(model.parameters()).device

    prefix = torch.tensor(vocab.encode(example), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        h_t, out = model(prefix)  # out: [1, seq_len, vocab]
        next_id = out[:, -1, :].argmax(-1).item() \

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if next_id == vocab.EOS_ID:
                break
            generated.append(vocab.id2symbol[next_id])

            inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            h_t, out = model(inp, h_0=h_t)
            next_id = out[:, -1, :].argmax(-1).item()

    result = ""

    if mode == "binary":


    return "".join(generated)