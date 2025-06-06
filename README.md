# 使用说明
## 问题一
在 ```mymodel.py```中实现了cnn代码框架，包括卷积，池化，不同的激活函数，batch_norm，dropout等操作。
具体见实验表格。

在```train.py```中集成了训练、测试和可视化流程。如果不训练，也可以根据注释加载模型测试已有模型。

注意：当前上传的模型架构以及模型文件是基本的对照模型，具体见论文。如许扩大模型结构或更改参数，请手动
调整 ```mymodel.py```

## 问题二
问题二中的几个主要问题，loss landscape，grad norm，以及difference。全部集成在```VGG_Loss_Landscape.py```中
并且是几个不同lr设置的统一执行。
