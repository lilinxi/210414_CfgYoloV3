#! /bin/bash

sftp lab2 << EOF
get -r /home/lenovo/data/lmf/210414_CfgYoloV3Sftp/logs/Voc_Test2Epoch83-Train_Loss0.0521-Val_Loss22.4719.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs
EOF
