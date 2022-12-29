# Cupy安装
请先检查自己的cuda版本，安装对应cupy。  
在conda环境中如下操作:  
`nvcc -V`  
`pip install pip cupy-cuda102`  
# 文件结构介绍  
`tool.py`:数据划分、归一化、one-hot编码  
`layer.py`:卷积层、池化层、激活函数、交叉熵、softmax等各种结构  
`model~model5`:五种卷积神经网络模型  
`train.py`:训练脚本  
`test.py`:测试脚本  
`train_label.py`、`train_data.py`:训练集及标签  
`test_data.py`:测试集  
