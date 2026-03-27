import argparse
from src.train.train_seq2seq import train_model
from src.evaluation.evaluate_seq2seq import *

parser = argparse.ArgumentParser()

parser.add_argument("--representation", type=str, default="binary", choices=["binary", "decimal"])
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--hid_dim", type=int, default=128)
parser.add_argument("--ttt_bottleneck_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--learning_rate", type=int, default=1e-3)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset_size", type=int, default=20000)
parser.add_argument("--num_digits", type=int, default=3)
parser.add_argument("--valid_samples", type=int, default=2000)
parser.add_argument("--test_num_digits", type=int, default=4)

args = parser.parse_args()

model, vocab, train_dataset, valid_dataset, valid_loader, device = train_model(
    representation=args.representation,
    num_digits=args.num_digits,
    train_samples=args.dataset_size,
    valid_samples=args.valid_samples,
    batch_size=args.batch_size,
    emb_dim=args.emb_dim,
    hidden_dim=args.hid_dim,
    ttt_bottleneck_dim=args.ttt_bottleneck_dim,
    epochs=args.epochs,
    lr=args.learning_rate,
    patience=args.patience,
    print_every=args.print_every
)

import torch
from src.models.seq2seq_gru import Seq2SeqGRUWithTTT

model = Seq2SeqGRUWithTTT(vocab=vocab)
state_dict = torch.load("outputs/best_model (4).pt", map_location="cpu")
model.load_state_dict(state_dict)

# print("Steps 5")
# evaluation(num_digits = args.test_num_digits, model= model, vocab= vocab, device= device, representation= args.representation)
# evaluation(num_digits = args.test_num_digits + 1, model= model, vocab= vocab, device= device, representation= args.representation)

print("Steps 5")
print("Inner loss for TTT: commutative")
evaluation(num_digits = args.test_num_digits, model= model, vocab= vocab, device= device, representation= args.representation,
           ttt_steps=10, inner_loss="commutative")
evaluation(num_digits = args.test_num_digits + 1, model= model, vocab= vocab, device= device, representation= args.representation,
           ttt_steps=10, inner_loss="commutative")


print("Steps 5")
print("Inner loss for TTT: zero_commutativity")
evaluation(num_digits = args.test_num_digits, model= model, vocab= vocab, device= device, representation= args.representation,
           ttt_steps=10, inner_loss="zero_commutativity")
evaluation(num_digits = args.test_num_digits + 1, model= model, vocab= vocab, device= device, representation= args.representation,
           ttt_steps=10, inner_loss="zero_commutativity")
