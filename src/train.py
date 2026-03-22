import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import *
from functools import partial
import torch.nn.functional as F
import json
import os

@torch.no_grad()
def evaluation(model, dataloader, loss_fn, vocab: Vocab):
  model.eval()

  losses = []
  accuracies = []
  exact_matches = []

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for inputs_id, targets in dataloader:
      inputs_id = inputs_id.to(device)
      targets = targets.to(device)

      _, outputs = model(inputs_id)
      outputs = outputs[:,:-1]
      targets = targets[:, 1:]

      loss = loss_fn(outputs.permute(0, 2, 1), targets)

      outputs = outputs.argmax(-1)
      mask = (targets != vocab.PAD_ID)
      corrects = (outputs == targets) & mask
      accuracy = corrects.sum().item() / mask.sum().item()

      accuracies.append(accuracy)
      losses.append(loss.item())

        #exact match
      pad_mask = (targets == vocab.PAD_ID)
      token_ok = (outputs == targets) | pad_mask
      exact_per_sample = token_ok.all(dim=1)
      exact_match = exact_per_sample.float().mean().item()
      exact_matches.append(exact_match)


  loss = np.mean(losses)
  accuracy = np.mean(accuracies)
  exact_match = np.mean(exact_matches)

  return loss, accuracy, exact_match

def train(model, dataloader, loss_fn, optimizer, vocab: Vocab):

    model.train()

    losses = []
    accuracies = []
    exact_matches = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for inputs_id, targets in dataloader:
        inputs_id = inputs_id.to(device)
        targets = targets.to(device)

        _, outputs = model(inputs_id)
        outputs = outputs[:,:-1]
        targets = targets[:, 1:]

        loss = loss_fn(outputs.permute(0, 2, 1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.argmax(-1)
        mask = (targets != vocab.PAD_ID)
        corrects = (outputs == targets) & mask
        accuracy = corrects.sum().item() / mask.sum().item()

        #accuracies.append((outputs.argmax(-1) == targets).float().mean().item())
        #accuracy
        accuracies.append(accuracy)
        losses.append(loss.item())

        #exact match
        pad_mask = (targets == vocab.PAD_ID)
        token_ok = (outputs == targets) | pad_mask
        exact_per_sample = token_ok.all(dim=1)
        exact_match = exact_per_sample.float().mean().item()
        exact_matches.append(exact_match)


    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    exact_match = np.mean(exact_matches)

    return loss, accuracy, exact_match

def train_commutative_function(model, dataloader, loss_fn, optimizer, vocab: Vocab,
                               lam:int = 0.05):

    model.train()

    losses = []
    accuracies = []
    exact_matches = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for inputs_id_ab, inputs_id_ba, targets in dataloader:
        inputs_id_ab = inputs_id_ab.to(device)
        inputs_id_ba = inputs_id_ba.to(device)
        targets = targets.to(device)
        targets = targets[:, 1:]

        _, outputs_ab = model(inputs_id_ab)
        outputs_ab = outputs_ab[:,:-1]
        loss_ab = loss_fn(outputs_ab.permute(0, 2, 1), targets)

        _, outputs_ba = model(inputs_id_ba)
        outputs_ba = outputs_ba[:,:-1]
        loss_ba = loss_fn(outputs_ba.permute(0, 2, 1), targets)

        logp_ab = F.log_softmax(outputs_ab, dim=-1)  # (B, T, V)
        logp_ba = F.log_softmax(outputs_ba, dim=-1)

        mask = (targets != vocab.PAD_ID)                  # (B, T)
        mask3 = mask.unsqueeze(-1).float()          # (B, T, 1)

        diff2 = (logp_ab - logp_ba) ** 2            # (B, T, V) MSE
        comm_loss = (diff2 * mask3).sum() / (mask3.sum() * outputs_ab.size(-1))
        loss = loss_ab + loss_ba + lam * comm_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs_ab.argmax(-1)
        mask = (targets != vocab.PAD_ID)
        corrects = (outputs == targets) & mask
        accuracy = corrects.sum().item() / mask.sum().item()

        #accuracies.append((outputs.argmax(-1) == targets).float().mean().item())
        #accuracy
        accuracies.append(accuracy)
        losses.append(loss.item())

        #exact match
        pad_mask = (targets == vocab.PAD_ID)
        token_ok = (outputs == targets) | pad_mask
        exact_per_sample = token_ok.all(dim=1)
        exact_match = exact_per_sample.float().mean().item()
        exact_matches.append(exact_match)


    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    exact_match = np.mean(exact_matches)

    return loss, accuracy, exact_match

def train_model(model, trainDataSet, testDataSet, vocab:Vocab, exp_name:str, trainCommativefunction: bool = False,
                    num_epochs: int = 200, patience: int = 50, min_delta: float = 1e-4):

    path = f"outputs/{exp_name}"
    os.makedirs(path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

    if trainCommativefunction:
      train_loader = DataLoader(trainDataSet, batch_size=128,
                                collate_fn=partial(collate_fn_ab_ba, vocab= vocab), shuffle=True)
    else:
      train_loader = DataLoader(trainDataSet, batch_size=128,
                                collate_fn=partial(collate_fn,vocab=vocab),shuffle=True)

    test_loader  = DataLoader(testDataSet,  batch_size=128,
                              collate_fn=partial(collate_fn, vocab= vocab), shuffle=True)

    best_val_loss = np.inf
    bad_epochs = 0

    for epoch in tqdm(range(1, num_epochs + 1)):
        if trainCommativefunction:
          train_loss, train_acc, train_em = train_commutative_function(model,
                                              train_loader, loss_fn, optimizer, vocab)
        else:
          train_loss, train_acc, train_em = train(model, train_loader, loss_fn, optimizer,
                                                  vocab)

        val_loss, val_acc, val_em = evaluation(model, test_loader, loss_fn, vocab)

        if epoch % 50 == 0 or epoch == 1:
            print()
            print(f"[{epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_em={train_em:.4f}")
            print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_em={val_em:.4f}")

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), f"outputs/{exp_name}/model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping: val_loss does not improve {patience} epochs. "
                  f"Best val_loss={best_val_loss:.4f}")
            break

    config = {
        "binary": model.binary,
        "dataset": len(trainDataSet),
        "epochs": 30,
        "hidden_size": model.hidden_size,
        "commutative_loss": trainCommativefunction
    }

    with open(f"outputs/{exp_name}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    model.load_state_dict(torch.load(f"outputs/{exp_name}/model.pt",
                                     map_location=device))
    return model