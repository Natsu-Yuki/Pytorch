if __name__ == '__main__':
    import torch as t
    import torch.utils.data as d
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    t.manual_seed(1)

    train_data = torchvision.datasets.MNIST(
        root=r'Data/mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    test_data = torchvision.datasets.MNIST(
        root=r'Data/mnist',
        train=False,
    )

    # print(type(train_data),type(test_data))

    train_loader = d.DataLoader(
        dataset=train_data,
        batch_size=50,
        shuffle=True
    )

    # plt.imshow(train_data.train_data.numpy()[0])
    # plt.title('{}'.format(train_data.train_labels[0]))
    # plt.show()


    # print(test_data.test_data.numpy().shape)

    test_X = test_data.test_data.unsqueeze(1)[:2000]/255
    test_y = test_data.test_labels[:2000]
    # print(test_X.numpy().shape, test_y.numpy().shape)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=1,  # input height
                    out_channels=16,  # n_filters
                    kernel_size=5,  # filter size
                    stride=1,  # filter movement/step
                    padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                ),  # output shape (16, 28, 28)
                nn.ReLU(),  # activation
                nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
            )
            self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
                nn.ReLU(),  # activation
                nn.MaxPool2d(2),  # output shape (32, 7, 7)
            )
            self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # print(x.size())
            x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
            output = self.out(x)
            return output

    cnn = CNN()

    # print(cnn)
    # cnn(train_data.train_data)
    # print(train_data.train_data.size())

    optimizer = t.optim.Adam(cnn.parameters(), lr=0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            # print(b_x.numpy().shape)
            output = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients             # cnn output
            if step% 10 == 0 :
                print(step)
            if step == 500:
                break

    test_output = cnn(test_X[:10])
    pred_y = t.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')