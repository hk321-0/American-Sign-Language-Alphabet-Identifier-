"""
What this file contains: 
1. Lab 4's reusable training loop
2. Controlled multi-seed comparisons 
3. Extracts CNN features for the CNN + XGBoost model.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train_model(model_fn, train_loader, val_loader, device,
                 max_epochs=30, patience=3, lr=1e-3, seed=0,
                 optimizer_name="adam"):
    #o/p best_model (whose weights are that of the best eval epoch) + history dict.
    set_seed(seed)
    model = model_fn().to(device)   #every run starts from a newly-initialized model since model_fn returns a fresh model

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_accuracies = [], []
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_acc = evaluate_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    history = {"train_losses": train_losses, "val_accuracies": val_accuracies,
               "best_val_acc": best_val_acc, "epochs_run": len(train_losses)}
    return model, history


def run_multi_seed(model_fn, train_loader, val_loader, test_loader, device,
                    seeds=(0, 1, 2), **train_kwargs):
    #running above single-seed loop across mulitiple seeds 
    #o/p mean and std of test accuracies + histories for each seed.
    test_accs = []
    histories = []
    for seed in seeds:
        model, history = train_model(model_fn, train_loader, val_loader, device,
                                      seed=seed, **train_kwargs)
        test_acc = evaluate_accuracy(model, test_loader, device)
        test_accs.append(test_acc)
        histories.append(history)
    test_accs = np.array(test_accs)
    return {
        "seeds": list(seeds),
        "test_accs": test_accs.tolist(),
        "mean": float(test_accs.mean()),
        "std": float(test_accs.std()),
        "histories": histories,
    }


@torch.no_grad()
def extract_cnn_embeddings(cnn_model, loader, device):
    ##i/p of XGBoost = embeddings and lebels extracted from CNN
    #obtained by running a loader through CNN 
    cnn_model.eval()
    feats, labels = [], []
    for X, y in loader:
        X = X.to(device)
        emb = cnn_model.extract_features(X)
        feats.append(emb.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)
