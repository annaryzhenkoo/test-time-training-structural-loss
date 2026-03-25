from src.data.dataset_generation import dataset_generation
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train.train_sequence_model import *
import argparse
from src.models.rnn import RNN
from src.models.gru import GRU

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_size", type=int, default=20000)
parser.add_argument("--mode", type=str, default="binary", choices=["binary", "decimal"])
parser.add_argument("--num_digits", type=int, default=3)
parser.add_argument("--commutative_loss", default=False)
parser.add_argument("--model", type=str, default="GRU")

args = parser.parse_args()

path = dataset_generation(args.dataset_size, args.mode, args.num_digits)

vocab = Vocab(mode = args.mode)

dataset = pd.read_csv(path)

train_dataset, test_dataset = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42
)

trainDataSet = DatasetSum(train_dataset.iloc[:,0].tolist())
train_loader = DataLoader(trainDataSet, batch_size=1,
                                collate_fn=partial(collate_fn,vocab=vocab),shuffle=False)


trainDataSet = DatasetSum(train_dataset.iloc[:,0].tolist())
testDataSet = DatasetSum(test_dataset.iloc[:,0].tolist())

if args.mode == "binary":
    binary = True
else:
    binary = False

if args.model == "GRU":
    model = GRU(vocab_size=len(vocab.symbol2id))
else:
    model = RNN(len(vocab.symbol2id), hidden_size=256, binary=binary)

train_model(model=model,trainDataSet=trainDataSet, testDataSet=testDataSet, vocab= vocab, num_epochs=500, exp_name="001",
                patience = 50, trainCommativefunction=args.commutative_loss)