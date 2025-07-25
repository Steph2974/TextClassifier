# TextClassifier

## 问题

1. 怎么只有两种分类？['正常-normal' '色情-sex word']

## 术语解释

1. Batch size 是指在训练过程中，每次迭代（iteration）用于计算损失函数和更新模型参数的数据样本数量
2. Data size 是指整个训练数据集的大小，即训练集中所有样本的总数
3. Epoch 是指整个训练数据集被完整地用于训练模型的次数
4. Dataset是数据集的抽象表示，它定义了如何访问数据集中的每个样本，一次返回一条记录
5. Dataloader用于从Dataset中批量加载数据，并提供多线程加载和数据打乱等功能
6. DataLoader作为数据流的来源，它会按照指定的批次大小和迭代次数来提供数据。这样模型就可以在每个epoch中迭代整个数据集，而不需要手动管理数据的加载和批处理

## 指令

export HF_HUB_ENABLE_HF_TRANSFER=1
export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;
huggingface-cli download --resume-download google-bert/bert-base-chinese --local-dir bert