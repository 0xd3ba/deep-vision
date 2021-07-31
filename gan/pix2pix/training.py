# training.py -- The trainer for the models

import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


###############################################
# The loss function: Binary Cross-Entropy Loss
LOSS_FN = nn.BCELoss

# The optimizer: Adam
# Learning Rate: 0.0002
# Not using any learning rate scheduler
LEARN_RATE = 0.0002
OPTIMIZER = optim.Adam
OPTIM_BETAS = (0.5, 0.999)

# Number of epochs to train the model: 20
N_EPOCHS = 20
###############################################


def train(train_loader, generator, discriminator, device):

    loss_fn = LOSS_FN()

    # NOTE: Because two models are being trained, need two optimizers for it
    gen_optimizer = OPTIMIZER(lr=LEARN_RATE, params=generator.parameters(), betas=OPTIM_BETAS)
    dis_optimizer = OPTIMIZER(lr=LEARN_RATE, params=discriminator.parameters(), betas=OPTIM_BETAS)

    generator.train()
    discriminator.train()

    generator.to(device)
    discriminator.to(device)

    # Start the training process
    # Run through the entire set for N_EPOCHS
    for epoch in range(N_EPOCHS):

        dis_epoch_loss = 0
        gen_epoch_loss = 0

        for batch_sketch, batch_src in tqdm.tqdm(train_loader):

            batch_sketch, batch_src = batch_sketch.to(device), batch_src.to(device)

            fake_batch_src = generator(batch_sketch)

            # PART-1: Train the Discriminator
            # -------------------------------------
            true_batch_y = torch.ones(batch_sketch.shape[0]).to(device)
            fake_batch_y = torch.zeros(batch_sketch.shape[0]).to(device)

            dis_pred_true = discriminator(batch_sketch, batch_src)          # Predictions on true data
            dis_pred_fake = discriminator(batch_sketch, fake_batch_src)     # Predictions on fake data

            dis_loss_true = loss_fn(dis_pred_true, true_batch_y)
            dis_loss_fake = loss_fn(dis_pred_fake, fake_batch_y)
            dis_loss = dis_loss_true + dis_loss_fake

            dis_epoch_loss += dis_loss.cpu().item()  # Simply track the loss for discriminator so far

            dis_optimizer.zero_grad()
            dis_loss.backward(retain_graph=True)    # Need to preserve the graph for "generator(noise_z)" forward pass
            dis_optimizer.step()

            # PART-2: Train the Generator
            # -------------------------------------
            dis_pred_fake = discriminator(batch_sketch, fake_batch_src)

            # The loss for generator also includes L1-distance between the prediction and the true target
            gen_loss = loss_fn(dis_pred_fake, true_batch_y) + F.l1_loss(fake_batch_src, batch_src)

            gen_epoch_loss += gen_loss.cpu().item()  # Simply tack the loss for generator so far

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

        print(f'Epoch[{epoch}]:  {dis_epoch_loss} (D)    {gen_epoch_loss} (G)')
