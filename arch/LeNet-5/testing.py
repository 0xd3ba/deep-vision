# testing.py -- The tester for the model

import tqdm
import torch.nn.functional as F


def get_labels(pred_scores):
    """ Returns the labels for the given scores """
    pred_probs = F.softmax(pred_scores, dim=-1)     # Output: (batch_size, 10)
    pred_labels = pred_probs.argmax(dim=-1)         # Output: (batch_size, )

    return pred_labels


def test(test_loader, model):
    """ Starts the testing of the model and reports the accuracy obtained """
    total_correct = 0
    total_samples = 0

    model.eval()

    for batch_x, batch_y in tqdm.tqdm(test_loader):
        pred_y_scores = model(batch_x)
        pred_labels = get_labels(pred_y_scores)

        n_correct = (pred_labels == batch_y).sum()
        total_correct += n_correct.item()
        total_samples += batch_y.shape[0]

    accuracy = round(total_correct * 100 / total_samples, 3)
    print(f'Correctly classified: {accuracy}% ({total_correct} / {total_samples})')
