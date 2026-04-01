import torch
from src.data.datasets_seq2seq import (
    AdditionDataset,
    build_vocab,
    decode_tokens,
    parse_tokens_to_int,
)

def continue_generation_after_EOS(model, representation: str, num_digits: int, max_len: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    vocab = build_vocab(representation)
    dataset = AdditionDataset(
        vocab=vocab,
        num_samples=20,
        num_digits=num_digits,
        representation=representation,
    )

    label_w = 16
    num_w = 12

    for i in range(len(dataset)):
        item = dataset[i]

        src_ids = item["src_ids"].unsqueeze(0).to(device)  # (1, S)
        src_lens = torch.tensor([len(item["src_ids"])], dtype=torch.long, device=device)

        with torch.no_grad():
            hidden = model.encode(src_ids, src_lens)   # (1, 1, H)
            hidden = model.ttt(hidden)                 # (1, 1, H)

            hidden_normal = hidden.clone()
            current = torch.tensor([[vocab.SOS_ID]], dtype=torch.long, device=device)
            pred_ids_normal = []
            eos_step_normal = None

            for step in range(max_len):
                logits, hidden_normal = model.decoder(current, hidden_normal)   # (1,1,V)
                step_logits = logits[:, -1, :]                                  # (1,V)
                next_id = int(step_logits.argmax(dim=-1).item())

                if next_id == vocab.EOS_ID:
                    eos_step_normal = step
                    break

                pred_ids_normal.append(next_id)
                current = torch.tensor([[next_id]], dtype=torch.long, device=device)

            hidden_override = hidden.clone()
            current = torch.tensor([[vocab.SOS_ID]], dtype=torch.long, device=device)
            pred_ids_override = []
            eos_steps_override = []

            for step in range(max_len):
                logits, hidden_override = model.decoder(current, hidden_override)  # (1,1,V)
                step_logits = logits[:, -1, :]                                     # (1,V)

                sorted_ids = torch.argsort(step_logits, dim=-1, descending=True)   # (1,V)
                top1_id = int(sorted_ids[0, 0].item())

                if top1_id == vocab.EOS_ID:
                    eos_steps_override.append(step)

                    chosen_id = None
                    for k in range(sorted_ids.size(1)):
                        cand = int(sorted_ids[0, k].item())
                        if cand != vocab.EOS_ID:
                            chosen_id = cand
                            break

                    if chosen_id is None:
                        break
                else:
                    chosen_id = top1_id

                pred_ids_override.append(chosen_id)
                current = torch.tensor([[chosen_id]], dtype=torch.long, device=device)

        pred_tokens_normal = decode_tokens(pred_ids_normal, vocab)
        pred_tokens_override = decode_tokens(pred_ids_override, vocab)

        pred_sum_normal = parse_tokens_to_int(pred_tokens_normal, representation)
        pred_sum_override = parse_tokens_to_int(pred_tokens_override, representation)
        true_sum = item["sum_"]

        print("=" * 90)
        print(f"Example {i + 1}")

        print("\nNUMBERS:")
        print(f"{'a:':<{label_w}}{item['a']:>{num_w}}")
        print(f"{'b:':<{label_w}}{item['b']:>{num_w}}")
        print(f"{'true sum:':<{label_w}}{true_sum:>{num_w}}")
        print(f"{'pred (EOS):':<{label_w}}{pred_sum_normal:>{num_w}}")
        print(f"{'pred (noEOS):':<{label_w}}{pred_sum_override:>{num_w}}")

        print("\nSEQUENCES (reversed):")
        print(f"{'input:':<{label_w}}{item['src_text']}")
        print(f"{'target:':<{label_w}}{item['tgt_text']}")
        print(f"{'pred (EOS):':<{label_w}}{''.join(pred_tokens_normal)}")
        print(f"{'pred (noEOS):':<{label_w}}{''.join(pred_tokens_override)}")

        print("\nINFO:")
        print(f"{'EOS step:':<{label_w}}{eos_step_normal}")
        print(f"{'override EOS at:':<{label_w}}{eos_steps_override}")