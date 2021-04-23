#! /bin/bash

sftp lab2 << EOF
get -r /home/lenovo/data/lmf/210414_CfgYoloV3Sftp/logs/Voc_Test5_Epoch100-Train_Loss4.5960-Val_Loss4.7334.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs
# get -r /home/lenovo/data/lmf/210414_CfgYoloV3Sftp_Cpy/logs/Voc_Test4Epoch259-Train_Loss4.2447-Val_Loss4.6763.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs
EOF
