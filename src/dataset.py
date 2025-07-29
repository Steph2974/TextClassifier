import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from config import MAX_LENGTH, MODEL_CONFIGS, CURRENT_MODEL

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer=None):
        self.data = pd.read_csv(data_path)
        # 获取当前模型的分词器类型
        tokenizer_class = MODEL_CONFIGS[CURRENT_MODEL]['tokenizer'] 
        model_name = MODEL_CONFIGS[CURRENT_MODEL]['name']
        
        if tokenizer_class == 'BertTokenizer':
            self.tokenizer = BertTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        elif tokenizer_class == 'AutoTokenizer':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_class}")
    
    def __len__(self):
        return len(self.data)

    # 获取单个样本
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['message'])
        label = 1 if self.data.iloc[idx]['label'] == '正常-normal' else 0
        # print(f"文本: {text}", type(text))  # 文本是字符串
        # print(f"标签: {label}", type(label))  # 标签是字符串
        
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt' # 返回 PyTorch 张量格式
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(), # 注意力掩码
            'labels': torch.tensor(label, dtype=torch.int8)
        }
    
# 帮我写一个例子，我想看看数据集类是否正常工作
if __name__ == "__main__":
    dataset = TextDataset('data/Taiwan_text_classification_dateset.csv')
    # 我想知道数据集中所有文本中的最大的10个长度，并标记他们的索引
    max_lengths = []
    for text in dataset.data['message']:
        max_lengths.append(len(text))
    max_lengths.sort(reverse=True)
    print(max_lengths[:10])
    print(dataset.data['message'][max_lengths[:10]])