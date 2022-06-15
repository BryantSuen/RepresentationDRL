pdf文件为作业报告

三个main_***.py文件独立运行，对应三个深度模型

models.py储存两个CNN，分别对应vanill/double DQN，和dueling DQN使用

memory.py实现了replay buffer
wrappers.py实现图像裁剪等预处理

如需运行线性模型，需要在models.py文件中，将DQN中的激活函数注释掉即可

videos文件夹中，子文件夹及子文件的命名方式如下：
A_video_B文件夹代表A模型在训练过程的B/3处，进行评估的结果
B_C.mp4视频文件代表B/3处的某模型，评估得到的游戏分数为C

## bisim部分
对应论文algorithm 1, 没有使用 actor-critic

参数
# hyperparameters for bisim
    DISCOUNT = 0.99
    BISIM_COEF = 0.5
    ENCODER_LR = 1e-3
    ENCODER_WEIGHT_DECAY = 0.
    DECODER_LR = 1e-3
    DECODER_WEIGHT_DECAY = 0.

TODO:bisim 的 save 与 load
