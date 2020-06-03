from datetime import datetime

import torch
from sklearn.metrics import f1_score

from real_or_not.testing import predict_on_model


def train_model(net, train_loader, val_loader, epochs, loss_function, optimizer, device):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f'{datetime.now()} epoch {epoch}')
        for i, (sentences, targets) in enumerate(train_loader):
            sentences = sentences.to(device)
            targets = targets.to(device)
            # targets = targets.type(torch.float)
            optimizer.zero_grad()

            tag_scores = net(sentences)
            tag_scores = tag_scores.squeeze()

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        train_actual, train_predicted = predict_on_model(net, train_loader, device)
        print(f'step: {epoch}')
        train_loss = f1_score(train_actual, train_predicted)
        train_losses.append(train_loss)
        print(f'train_f1: {train_loss}')
        if val_loader is not None:
            val_actual, val_predicted = predict_on_model(net, val_loader, device)
            val_loss = f1_score(val_actual, val_predicted)
            val_losses.append(val_loss)
            print(f'val_f1: {val_loss}')
    print(f'max_train_f1: {max(train_losses)}')
    if val_loader is not None:
        print(f'max_val_f1: {max(val_losses)}')
