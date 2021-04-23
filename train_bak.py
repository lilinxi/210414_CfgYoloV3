import numpy

import torch.utils.data.dataloader

import conf.config
import model.yolov3net, model.yolov3loss
import dataset.voc_dataset
import train_utils

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.set_device(1)

    # 0. 确保每次的伪随机数相同以便于问题的复现
    numpy.random.seed(0)
    torch.manual_seed(1)

    # 1. 训练参数
    Config = conf.config.VocConfig

    print("config:\n", Config)

    # 提示 OOM 或者显存不足请调小 Batch_size
    Freeze_Train_Batch_Size = 16
    Freeze_Eval_Batch_Size = 16

    Unfreeze_Train_Batch_Size = 8
    Unfreeze_Eval_Batch_Size = 8

    Init_Epoch = 0  # 起始世代
    Freeze_Epoch = 50  # 冻结训练的世代
    Unfreeze_Epoch = 2000  # 总训练世代

    Freeze_Epoch_LR = 1e-3
    Unfreeze_Epoch_LR = 1e-4

    Freeze_Epoch_Gamma = 0.96  # 0.96 ^ 50 = 0.12988
    Unfreeze_Epoch_Gamma = 0.98  # 0.98^100 = 0.13262

    Num_Workers = 12
    Suffle = True

    Test_Name = "Voc_Test7"

    # 2. 创建 yolo 模型，训练前一定要修改 Config 里面的 classes 参数，训练的是 YoloNet 不是 Yolo
    yolov3_net = model.yolov3net.YoloV3Net(Config)

    # 3. 加载 darknet53 的权值作为预训练权值
    train_utils.load_pretrained_weights(yolov3_net, Config["pretrained_weights_path"], Config["cuda"])

    # 4. 开启训练模式
    yolov3_net = yolov3_net.train()

    if Config["cuda"]:
        yolov3_net = yolov3_net.cuda()

    print("yolov3_net in cuda") if Config["cuda"] else print("yolov3_net not in cuda")

    # 5. 建立 loss 函数
    yolov3_loss = model.yolov3loss.YoloV3Loss(Config)

    # 6. 加载训练数据集和测试数据集
    freeze_train_data_loader = dataset.voc_dataset.VOCDataset.TrainDataloader(
        config=Config,
        batch_size=Freeze_Train_Batch_Size,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    freeze_train_batch_num = len(freeze_train_data_loader)

    freeze_validate_data_loader = dataset.voc_dataset.VOCDataset.EvalAsTrainDataloader(
        config=Config,
        batch_size=Freeze_Eval_Batch_Size,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    freeze_validate_batch_num = len(freeze_validate_data_loader)

    unfreeze_train_data_loader = dataset.voc_dataset.VOCDataset.TrainDataloader(
        config=Config,
        batch_size=Unfreeze_Train_Batch_Size,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    unfreeze_train_batch_num = len(unfreeze_train_data_loader)

    unfreeze_validate_data_loader = dataset.voc_dataset.VOCDataset.EvalAsTrainDataloader(
        config=Config,
        batch_size=Unfreeze_Eval_Batch_Size,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    unfreeze_validate_batch_num = len(unfreeze_validate_data_loader)

    # 7. 粗略训练预测头

    # 7.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Freeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 7.2 冻结特征网络
    for param in yolov3_net.backbone.parameters():
        param.requires_grad = False

    # 7.3 训练若干 Epoch
    for epoch in range(Init_Epoch, Freeze_Epoch):
        train_utils.train_one_epoch(
            Test_Name,
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            freeze_train_batch_num,  # 训练集批次数
            freeze_validate_batch_num,  # 验证集批次数
            Freeze_Epoch,  # 总批次
            freeze_train_data_loader,  # 训练集
            freeze_validate_data_loader,  # 验证集
            Config["cuda"],
        )
        lr_scheduler.step()  # 更新步长

    # 8. 精细训练预测头和特征网络

    # 8.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Unfreeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    # 8.2 解冻特征网络
    for param in yolov3_net.backbone.parameters():
        param.requires_grad = True

    # 8.3 训练若干 Epoch
    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        train_utils.train_one_epoch(
            Test_Name,
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            unfreeze_train_batch_num,  # 训练集批次数
            unfreeze_validate_batch_num,  # 验证集批次数
            Unfreeze_Epoch,  # 总批次
            unfreeze_train_data_loader,  # 训练集
            unfreeze_validate_data_loader,  # 验证集
            Config["cuda"],
        )
        lr_scheduler.step()  # 更新步长
