# TextClassifier

## 问题

1. 怎么只有两种分类？['正常-normal' '色情-sex word']
2. bert可以把训练批次设置为64，'BATCH_SIZE = 64'，但是bge不行，会内存不够。
3. 警告：警告信息是由于 Hugging Face 的 tokenizers 在进程被 fork 后，启用了并行处理。为了避免潜在的死锁，它会自动禁用并行处理。
解决方法：在运行 Python 脚本之前，在命令行中设置环境变量 export TOKENIZERS_PARALLELISM=false
4. 很慢。比如 Epoch 1/15:   1%|     | 5/769 [07:51<19:59:50, 94.23s/it, Loss=1.9651, Avg_Loss=1.2995, Batch=5/769、Epoch 1/15:   0%|   | 6/3073 [04:57<40:22:19, 47.39s/it, Loss=1.2613, Avg_Loss=1.2264, Batch=6/3073]
5. 没有训练效果。损失函数没有变化、准确率没有变化。应该也是选错了模型。
6. 纹丝不动。
Epoch 1/15: 100%|████| 769/769 [02:57<00:00,  4.32it/s, Loss=0.1653, Avg_Loss=0.5192, Batch=769/769]
Epoch 2/15: 100%|████| 769/769 [02:56<00:00,  4.37it/s, Loss=0.1380, Avg_Loss=0.5186, Batch=769/769]
Epoch 3/15: 100%|████| 769/769 [02:56<00:00,  4.37it/s, Loss=0.2650, Avg_Loss=0.5167, Batch=769/769]
Epoch 4/15:  40%|█▌  | 306/769 [01:11<01:26,  5.37it/s, Loss=0.8758, Avg_Loss=0.5212, Batch=306/769]


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

huggingface-cli download --resume-download Qwen/Qwen3-4B-Base --local-dir qwen 

通过设置环境变量 TOKENIZERS_PARALLELISM 来禁用并行处理：
export TOKENIZERS_PARALLELISM=false