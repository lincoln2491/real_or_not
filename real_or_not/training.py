from datetime import datetime

from sklearn.metrics import f1_score

from real_or_not.testing import predict_on_model


def train_model(net, train_loader, val_loader, epochs, loss_function, optimizer, batch_size, device, train_ds):
    for epoch in range(epochs):
        print(f'{datetime.now()}: epoch {epoch}')
        for i, (sentences, targets) in enumerate(train_loader):
            if i % 1000 == 0:
                print(f'{datetime.now()}: {((i * batch_size) / len(train_ds)) * 100}')
            sentences = sentences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            tag_scores = net(sentences)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        train_actual, train_predicted = predict_on_model(net, train_loader, device)
        val_actual, val_predicted = predict_on_model(net, val_loader, device)
        print(
            f'epoch: {epoch}: train loss {f1_score(train_actual, train_predicted)} val loss {f1_score(val_actual, val_predicted)}')



