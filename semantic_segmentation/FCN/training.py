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

            x, y = dataset[rand_indices[i]]

            # Because we are taking items directly from the dataset class, they are
            # not provided as a batch, i.e. there is no batch dimension. Need to add one for both
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            pred_y_scores = model(x)
            batch_loss = loss_fn(pred_y_scores, y)

            # Calculate the gradient and back-propagate the loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.cpu().item()

        print(f'Epoch[{epoch}]:  {epoch_loss}')
