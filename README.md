# transformer : attention is all you need
环境搭建：

`conda create -n transformer_env python=3.8`

激活环境：

`conda activate transformer_env`

进入环境：

`conda activate transformer_env`

下载相关的包：

`pip install -e .`

下载对应的语言包：

如果你的包的版本跟我的一样，我已经将需要的gz文件放在`transformer->model`下可以直接使用下面的命令安装。

`pip install transformer/model/en_core_web_sm-3.1.0.tar.gz`

`pip install transformer/model/zh_core_web_sm-3.1.0.tar.gz`

快速体验：(注：这里为了计算快epoch=2，可以通过增加epoch来优化训练效果)

`cd transformer`

`python train.py`

--------------------------

具体讲解见：https://blog.csdn.net/m0_47719040/article/details/150608889?spm=1001.2014.3001.5502
