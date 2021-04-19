#! /bin/bash

sftp lab2 << EOF
get -r /home/lenovo/data/lmf/210414_CfgYoloV3Sftp/logs/Voc_Test3Epoch354-Train_Loss4.2421-Val_Loss4.6922.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs
get -r /home/lenovo/data/lmf/210414_CfgYoloV3Sftp_Cpy/logs/Voc_Test4Epoch259-Train_Loss4.2447-Val_Loss4.6763.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs
EOF
