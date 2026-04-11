import argparse
from src.train.train_seq2seq import train_model
from src.train.train_seq2seq_with_carry import train_model_with_carry
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
parser.add_argument("--with_carry", action="store_true", default=True)
parser.add_argument("--carry_emb_dim", type=int, default=4)
parser.add_argument("--carry_loss_weight", type=float, default=1.0)

args = parser.parse_args()

if args.with_carry:
    checkpoint_path = "outputs/seq2seq_with_gru_and_carry_.pt"

    print("TRAIN WITH CARRY")
    model, vocab, carry_vocab, train_dataset, valid_dataset, valid_loader, device = train_model_with_carry(
        representation=args.representation,
        num_digits=args.num_digits,
        train_samples=args.dataset_size,
        valid_samples=args.valid_samples,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hid_dim,
        ttt_bottleneck_dim=args.ttt_bottleneck_dim,
        carry_emb_dim=args.carry_emb_dim,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        print_every=args.print_every,
        checkpoint_path=checkpoint_path,
        carry_loss_weight=args.carry_loss_weight,
    )
else:
    checkpoint_path = "outputs/seq2seq_with_gru.pt"

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
        print_every=args.print_every,
        checkpoint_path=checkpoint_path,
    )

model = model.to("cpu")

show_predictions(model, vocab, num_digits=3, device="cpu", n=5, max_len=34)
show_predictions(model, vocab, num_digits=4, device="cpu", n=5, max_len=34)

simple_evaluation(
    num_digits=args.test_num_digits,
    model=model,
    vocab=vocab,
    device="cpu",
    representation=args.representation,
)

simple_evaluation(
    num_digits=args.num_digits,
    model=model,
    vocab=vocab,
    device="cpu",
    representation=args.representation,
)