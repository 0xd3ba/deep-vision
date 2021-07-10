# training.py -- The trainer for the model

import tqdm
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
###############################################


def train(train_loader, model, device):
    loss_fn = LOSS_FN()
    optimizer = OPTIMIZER(lr=LEARN_RATE, params=model.parameters())

    model.train()
    model.to(device)

    # Start the training process
    # Run through the entire set for N_EPOCHS
    for epoch in range(N_EPOCHS):

        epoch_loss = 0

        for batch_x, batch_y in tqdm.tqdm(train_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_y_scores = model(batch_x)
            batch_loss = loss_fn(pred_y_scores, batch_y)

            # Calculate the gradient and back-propagate the loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.cpu().item()

        print(f'Epoch[{epoch}]:  {epoch_loss}')
