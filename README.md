# 210414_CfgYoloV3

1. Voc_Test1: 
    - 损失权重都为 1
    - 无 ignore 损失
    - 无数据增强变换
    - 训练参数：
        - batch_size：16
        - freeze_epoch：20
2. Voc_Test2: 
    - 损失权重为 1，class 损失权重为 5
    - 无 ignore 损失
    - 无数据增强变换
    - 训练参数：
        - batch_size：16
        - freeze_epoch：20
3. Voc_Test3: 
    - 损失权重都为 1
    - 无 ignore 损失
    - 有数据增强变换
        - 缩放变换
        - 色域变换
        - 翻转变换
    - 训练参数：
        - batch_size：64
        - freeze_epoch：50