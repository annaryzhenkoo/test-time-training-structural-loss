import argparse
from src.train.train_seq2seq import train_model
from src.evaluation.evaluate_seq2seq import *
from src.train.inference_seq2seq import show_predictions


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

#state_dict = torch.load("outputs/best_model (4).pt", map_location="cpu")
#model.load_state_dict(state_dict)

#model = model.to("cpu")

show_predictions(model, vocab, num_digits=3, device=device, n=5, max_len=34)

show_predictions(model, vocab, num_digits=4, device=device, n=5, max_len=34)

evaluation(num_digits = args.num_digits, model= model, vocab= vocab, device= device, representation= args.representation)
evaluation(num_digits = args.test_num_digits, model= model, vocab= vocab, device= device, representation= args.representation)


