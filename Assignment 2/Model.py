import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2


class Autoencoder:
    def __init__(self, param):
        self.param = param
        self.encoder = Encoder()
        self.decoder = Decoder()

    def fit(self, dataloader):
        pass

    def get_encoder(self):
        return self.encoder


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(1, 5, kernel_size=(3, 3))
        # self.conv_2 = nn.Conv2d(256, 128, kernel_size=(5, 5))
        # self.conv_3 = nn.Conv2d(128, 8, kernel_size=(5, 5))
        self.lv = nn.Linear(3380, 10)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv_1(x))
        # print(x.shape)
        # x = F.relu(self.conv_2(x))
        # print(x.shape)
        # x = F.relu(self.conv_3(x))
        # print(x.shape)
        x = torch.reshape(x, shape=(1, 1, 1, -1))
        # print(x.shape)
        x = self.lv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Linear(10, 784)
        # self.conv_2=nn.Linear(256, 512)
        # self.conv_3=nn.Linear(512, 784)
        # self.conv_1 = nn.ConvTranspose2d(1, 128, 5)
        # self.conv_2 = nn.ConvTranspose2d(128, 256, 5)
        # self.conv_3 = nn.ConvTranspose2d(256, 1, 3)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        # x = F.relu(self.conv_2(x))
        # x = F.relu(self.conv_3(x))
        x = torch.reshape(x, shape=(1, 28, 28))
        return x


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor()])
    cifar_data = tv.datasets.MNIST("./Data", download=False, transform=transform)
    print(cifar_data[0][0])
    # for x in range(20):
    #     cv2.imshow("test"+str(x), cifar_data[x][0].numpy().reshape(28, 28, 1))
    #     cv2.waitKey()

    # cifar_data=np.ndarray(cifar_data[:20])
    data_loader = torch.utils.data.DataLoader(cifar_data, batch_size=64, shuffle=False, num_workers=8)

    encoder = Encoder()
    decoder = Decoder()

    criterion = nn.MSELoss()
    param = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.8)

    for epoch in range(10000):
        print("New epoch", epoch)
        for num, (local_batch, local_labels) in enumerate(cifar_data):
            # if num > 10: break
            # print(local_batch)
            # print(local_labels)
            encoder.zero_grad()
            decoder.zero_grad()

            local_batch = local_batch.reshape(-1, 1, 28, 28)

            out_encoder = encoder(local_batch)

            # print("out_encoder", out_encoder.shape)

            output = decoder(out_encoder)

            # print("out_decoder", output.shape)
            loss = criterion(output, local_batch)
            loss.backward()
            optimizer.step()

            # for x in decoder.parameters():
            #     print(x.grad)

            # print(loss)
            # print(output.shape)
            if num % 5000 == 0:
                img = output.detach().numpy()[0, :, :]
                print(img.shape)
                cv2.imshow(str(epoch) + "inp", local_batch.numpy().reshape(28, 28, 1))
                cv2.imshow(str(epoch), img.reshape(28, 28, 1))
                cv2.waitKey(delay=1)
                # plt.imshow(img)
                # cv2.waitKey()
