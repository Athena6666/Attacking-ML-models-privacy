import models
from train import *
import torch
import torchvision 
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

def main():

    lr = 0.001 # learning rate 学习率：学习率决定了每次更新参数时的步长大小。它控制着参数的更新速度。较大的学习率可能导致参数在更新过程中跳过最优解，而较小的学习率可能导致收敛速度过慢。
    batch_size = 4
    k=3
    n_epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target_net_type = models.mlleaks_cnn
    shadow_net_type = models.mlleaks_cnn

    # 数据预处理，利用torchvision.transforms模块提供了一些预定义的图像转换函数
    train_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Pad(2),
        

        #torchvision.transforms.RandomRotation(10),
        #torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        
        # 将PIL图像或NumPy数组转换为PyTorch Tensor格式
        torchvision.transforms.ToTensor(),

        # 对输入的张量进行归一化操作。这里的参数表示数据集的均值和标准差，用于将输入数据的每个通道进行归一化，使其均值为 0、标准差为 1。这种归一化通常有助于提高模型的鲁棒性和收敛速度。
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 通过将这些转换操作放入 torchvision.transforms.Compose([]) 中，可以按照所列顺序依次应用这些操作，从而构建一个完整的数据预处理管道
    test_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load training set  
    '''
    通过 torchvision.datasets.FashionMNIST 创建了一个 FashionMNIST 数据集的实例
    FashionMNIST 是一个包含服装类别的图像数据集，用于图像分类任务。其中的参数如下：
    1. train=True：表示创建的是训练集。如果设置为 False，则创建测试集。
    2. transform=train_transform：指定了之前定义的数据预处理转换操作集合 train_transform，用于对图像进行预处理。
    3. download=True：如果数据集尚未下载到指定路径（'../../Datasets/'），则自动下载。
    '''
    trainset = torchvision.datasets.FashionMNIST('../../Datasets/', train=True, transform=train_transform, download=True)
    '''
    通过 torch.utils.data.DataLoader 创建了一个训练数据加载器 trainloader。
    数据加载器可以将数据集分批次地提供给模型进行训练，以利用批处理的并行计算能力。其中的参数如下：
    1. trainset：指定要加载的数据集对象。
    2. batch_size：指定每个批次的样本数量。
    3. shuffle=True：表示在每个 epoch（训练循环）开始时打乱数据集的顺序，以增加样本间的随机性。
    4. num_workers=2：指定用于数据加载的工作线程数。
    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # load test set 
    testset = torchvision.datasets.FashionMNIST('../../Datasets/', train=False, transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    # 根据总样本数，将训练集划分为四个子集
    total_size = len(trainset)
    split1 = total_size // 4
    split2 = split1*2
    split3 = split1*3

    indices = list(range(total_size))
    shadow_train_idx = indices[:split1]
    shadow_out_idx = indices[split1:split2]
    target_train_idx = indices[split2:split3]
    target_out_idx = indices[split3:]

    # 创建一个能够从给定索引列表中随机选择样本的采样器
    shadow_train_sampler = SubsetRandomSampler(shadow_train_idx)
    shadow_out_sampler = SubsetRandomSampler(shadow_out_idx)
    target_train_sampler = SubsetRandomSampler(target_train_idx)
    target_out_sampler = SubsetRandomSampler(target_out_idx)

    shadow_train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=shadow_train_sampler, num_workers=1)
    shadow_out_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=shadow_out_sampler, num_workers=1)
    target_train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=target_train_sampler, num_workers=1)
    target_out_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=target_out_sampler, num_workers=1)

    '''
    首先，代码通过target_net_type()创建了一个目标模型target_net，并将其移到到指定设备上
    然后，通过target_net.apply(models.weights_init)应用了一个初始化函数models.weights_init，用于初始化目标模型的权重
    接着，用nn.CrossEntropyLoss()定义了目标模型的损失函数target_loss，用于计算模型在训练中的损失
    最后，使用optim.Adam(target_net.parameters(), lr=lr) 创建了一个Adam优化器 target_optim，用于更新目标模型的参数。
    '''
    target_net = target_net_type().to(device)
    target_net.apply(models.weights_init)
    target_loss = nn.CrossEntropyLoss()
    target_optim = optim.Adam(target_net.parameters(), lr=lr)

    shadow_net = shadow_net_type().to(device)
    shadow_net.apply(models.weights_init)
    shadow_loss = nn.CrossEntropyLoss()
    shadow_optim = optim.Adam(shadow_net.parameters(), lr=lr)

    attack_net = models.mlleaks_mlp(n_in=k).to(device)
    attack_net.apply(models.weights_init)
    attack_loss = nn.BCELoss() #二进制交叉熵损失函数，意味着攻击模型的任务是执行二进制分类，例如对目标模型进行攻击时判断样本是否属于特定类别
    attack_optim = optim.Adam(attack_net.parameters(), lr=lr) # Adam 是一种常用的优化算法，用于调整模型参数以最小化损失函数


    train(shadow_net, shadow_train_loader, testloader, shadow_optim, shadow_loss, n_epochs, classes=classes)
    train_attacker(attack_net, shadow_net, shadow_train_loader, shadow_out_loader, attack_optim, attack_loss, n_epochs=1, k=k)
    train(target_net, target_train_loader, testloader, target_optim, target_loss, n_epochs, classes=classes)
    eval_attack_net(attack_net, target_net, target_train_loader, target_out_loader, k)

if __name__ == '__main__':
    main()