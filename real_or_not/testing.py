import torch
import numpy as np


def predict_on_model(net, dataloader, device, return_probability=False):
    actual = []
    predicted = []
    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, list):
                sentences, targets = data
                actual.extend(targets.numpy())
            else:
                sentences = data
            sentences = sentences.to(device)
            tag_scores = net(sentences)
            predicted.extend(tag_scores.cpu().numpy())
    predicted = np.array(predicted)
    if not return_probability:
        predicted = predicted.argmax(axis=1)
    return actual, predicted
