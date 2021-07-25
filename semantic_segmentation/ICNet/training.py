# training.py -- The trainer for the model

import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim


###############################################
# The loss function: Cross-Entropy Loss
LOSS_FN = nn.CrossEntropyLoss

# The optimizer: Adam
# Learning Rate: 0.001
# Not using any learning rate scheduler
LEARN_RATE = 0.001
OPTIMIZER = optim.Adam

# Number of epochs to train the model: 20
N_EPOCHS = 20

# Random seed for reproducibility
RANDOM_SEED = 42
###############################################


def train(dataset, model, device):

    np.random.seed(RANDOM_SEED)

    loss_fn = LOSS_FN()
    optimizer = OPTIMIZER(lr=LEARN_RATE, params=model.parameters())

    model.train()
    model.to(device)

    # Start the training process
    # Run through the entire set for N_EPOCHS
    for epoch in range(N_EPOCHS):

        epoch_loss = 0
        rand_indices = np.random.permutation(len(dataset))

        for i in tqdm.tqdm(range(rand_indices.shape[0])):

            (x1, y1), (x2, y2), (x3, y3) = dataset[rand_indices[i]]

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            y_pred_scores_1, y_pred_scores_2, y_pred_scores_3 = model(x1, x2, x3)

            y_pred_1_loss = loss_fn(y_pred_scores_1, y1)
            y_pred_2_loss = loss_fn(y_pred_scores_2, y2)
            y_pred_3_loss = loss_fn(y_pred_scores_3, y3)

            # Give weights to the losses
            y_weighted_loss = 0.4*y_pred_1_loss + 0.3*y_pred_2_loss + 0.3*y_pred_3_loss

            # Calculate the gradient and back-propagate the loss
            optimizer.zero_grad()
            y_weighted_loss.backward()
            optimizer.step()

            epoch_loss += y_weighted_loss.cpu().item()

        print(f'Epoch[{epoch}]:  {epoch_loss}')
