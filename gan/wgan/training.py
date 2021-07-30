# training.py -- The trainer for the models

import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from loss import WGANLossFunction


###############################################
# The loss function: Custom loss function for
# WGAN
LOSS_FN = WGANLossFunction

# The optimizer: RMSProp
# Learning Rate: 0.00005
# Not using any learning rate scheduler
LEARN_RATE = 5e-5
OPTIMIZER = optim.RMSprop

GRAD_CLIP = 0.01      # The gradient of critic is clipped to [-GRAD_CLIP, GRAD_CLIP]
N_CRITIC_UPDATES = 5  # Number of times the critic is trained before generator is trained

# Number of epochs to train the model: 20
N_EPOCHS = 20
###############################################


def train(train_loader, generator, critic, device):

    loss_fn = LOSS_FN()

    # NOTE: Because two models are being trained, need two optimizers for it
    gen_optimizer = OPTIMIZER(lr=LEARN_RATE, params=generator.parameters())
    critic_optimizer = OPTIMIZER(lr=LEARN_RATE, params=critic.parameters())

    generator.train()
    critic.train()

    generator.to(device)
    critic.to(device)

    # Start the training process
    # Run through the entire set for N_EPOCHS
    for epoch in range(N_EPOCHS):

        critic_epoch_loss = 0
        gen_epoch_loss = 0

        for batch_x in tqdm.tqdm(train_loader):

            batch_x = batch_x.to(device)
            fake_batch_x = None

            # PART-1: Train the critic for N_CRITIC_UPDATES
            # before training the generator once
            # -------------------------------------------------
            for _ in range(N_CRITIC_UPDATES):
                noise_z = torch.randn(batch_x.shape[0], generator.in_dims).to(device)
                fake_batch_x = generator(noise_z)

                critic_pred_true = critic(batch_x)          # Predictions on true data
                critic_pred_fake = critic(fake_batch_x)     # Predictions on fake data

                critic_loss = loss_fn(fake_pred=critic_pred_fake, real_pred=critic_pred_true)

                critic_epoch_loss += critic_loss.cpu().item()  # Simply track the loss for critic so far

                critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)   # Need to preserve the graph for "generator(noise_z)" forward pass

                # Clamp the gradients
                for param in critic.parameters():
                    param.data = param.data.clamp(min=-GRAD_CLIP, max=GRAD_CLIP)

                critic_optimizer.step()

            # PART-2: Train the Generator
            # -------------------------------------
            critic_pred_fake = critic(fake_batch_x)
            gen_loss = loss_fn(fake_pred=critic_pred_fake)

            gen_epoch_loss += gen_loss.cpu().item()  # Simply tack the loss for generator so far

            gen_optimizer.zero_grad()
            gen_loss.backward()

            # Generator doesn't need to clamp its gradients, unlike critic
            gen_optimizer.step()

        print(f'Epoch[{epoch}]:  {critic_epoch_loss} (C)    {gen_epoch_loss} (G)')
