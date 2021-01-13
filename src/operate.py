import itertools
from tqdm import tqdm
from dataset import DATA_SHAPE
import numpy as np
from util import *
from model import init_weights
import matplotlib.pyplot as plt


def train(model, dl_A, dl_B, device, params):
    model.D_A.to(device)
    model.G_AB.to(device)
    model.D_B.to(device)
    model.G_BA.to(device)

    model.D_A.apply(init_weights)
    model.G_AB.apply(init_weights)
    model.D_B.apply(init_weights)
    model.G_BA.apply(init_weights)

    optim_G = torch.optim.Adam(
        itertools.chain(model.G_AB.parameters(), model.G_BA.parameters()),
        lr=params["lr"], betas=params["adam_betas"])
    optim_D_A = torch.optim.Adam(model.D_A.parameters(), lr=params["lr"], betas=params["adam_betas"])
    optim_D_B = torch.optim.Adam(model.D_B.parameters(), lr=params["lr"], betas=params["adam_betas"])

    criterion_gan = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    criterion_gan.to(device)
    cycle_loss.to(device)
    criterion_identity.to(device)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G,
                                                       lr_lambda=LambdaLR(params["epochs"], 0,
                                                                          params["decay_epoch"]).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_D_A, lr_lambda=LambdaLR(params["epochs"], 0,
                                                                                       params["decay_epoch"]).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_D_B, lr_lambda=LambdaLR(params["epochs"], 0,
                                                                                       params["decay_epoch"]).step)

    input_A = torch.cuda.FloatTensor(int(params["batch_size"] / 2), DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2]).to(
        device)
    input_B = torch.cuda.FloatTensor(int(params["batch_size"] / 2), DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2]).to(
        device)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # TODO: LOGGER
    for epoch in tqdm(range(params["epochs"])):
        for i, batch in enumerate(zip(dl_A, dl_B)):
            torch.cuda.empty_cache()

            batch_A, batch_B = batch
            try:
                real_A = Variable(input_A.copy_(batch_A))
                real_B = Variable(input_B.copy_(batch_B))
            except RuntimeError:
                continue

            patchGAN_output_shape = (1, DATA_SHAPE[1] // (2 ** 4), DATA_SHAPE[2] // (2 ** 4))
            target_real = Variable(torch.cuda.FloatTensor(np.ones((real_A.size(0), *patchGAN_output_shape))),
                                   requires_grad=False).to(device)
            target_fake = Variable(torch.cuda.FloatTensor(np.zeros((real_A.size(0), *patchGAN_output_shape))),
                                   requires_grad=False).to(device)

            optim_G.zero_grad()
            model.G_AB.train()
            model.G_BA.train()

            # Identity loss
            loss_identity_A = criterion_identity(model.G_BA(real_A), real_A)
            loss_identity_B = criterion_identity(model.G_AB(real_B), real_B)

            loss_identity = (loss_identity_A + loss_identity_B) / 2

            # GAN loss
            fake_B = model.G_AB(real_A)
            loss_GAN_AB = criterion_gan(model.D_B(fake_B), target_real)
            fake_A = model.G_BA(real_B)
            loss_GAN_BA = criterion_gan(model.D_A(fake_A), target_real)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recovered_A = model.G_BA(fake_B)
            loss_cycle_ABA = cycle_loss(recovered_A, real_A)
            recovered_B = model.G_AB(fake_A)
            loss_cycle_BAB = cycle_loss(recovered_B, real_B)

            loss_cycle = (loss_cycle_ABA + loss_cycle_BAB) / 2

            # Total loss
            loss_G = loss_GAN + 15 * loss_cycle + 10 * loss_identity

            loss_G.backward()
            optim_G.step()

            ###### Discriminator A ######
            optim_D_A.zero_grad()

            # Real loss
            loss_D_real = criterion_gan(model.D_A(real_A), target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            loss_D_fake = criterion_gan(model.D_A(fake_A.detach()), target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) / 2

            loss_D_A.backward()
            optim_D_A.step()

            ###### Discriminator B ######
            optim_D_B.zero_grad()

            # Real loss
            loss_D_real = criterion_gan(model.D_B(real_B), target_real)

            # Fake loss
            fake_B = fake_A_buffer.push_and_pop(fake_B)
            loss_D_fake = criterion_gan(model.D_B(fake_B.detach()), target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) / 2

            loss_D_B.backward()
            optim_D_B.step()

            print("GAN loss:", loss_G.data.item(), " | Disc_A loss: ", loss_D_A.item(), " | Disc_B loss: ",
                  loss_D_B.item())

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Plot last batch of epoch and its corresponding conversions.
        model.G_AB.eval()
        model.G_BA.eval()
        converted = list()
        converted.append((tensor_to_image(real_A[0]), tensor_to_image(model.G_AB(real_A)[0])))
        converted.append((tensor_to_image(real_B[0]), tensor_to_image(model.G_BA(real_B)[0])))
        plot_output(converted)


def convert(model, device, dl_A=None, dl_B=None):
    converted = list()
    if dl_A is not None and dl_B is None:
        model.G_AB.eval()
        input_A = torch.cuda.FloatTensor(1, DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2]).to(device)
        for img_A in dl_A:
            real_A = Variable(input_A.copy_(img_A))
            converted.append((tensor_to_image(real_A), tensor_to_image(model.G_AB(real_A))))
    elif dl_A is None and dl_B is not None:
        model.G_BA.eval()
        input_B = torch.cuda.FloatTensor(1, DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2]).to(device)
        for img_B in dl_B:
            real_B = Variable(input_B.copy_(img_B))
            converted.append((tensor_to_image(real_B), tensor_to_image(model.G_BA(real_B))))
    else:
        raise Exception
    return converted
