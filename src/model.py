import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CONFIGS, CURRENT_MODEL, NUM_CLASSES

class BertClassifier(nn.Module):
    def __init__(self, hidden_size=512):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_CONFIGS['bert']['name'])
        # self.bert.requires_grad_(False)  # 冻结 BERT 参数
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),  # 768 -> 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)  # 256 -> 2（二分类）
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits

class BgeClassifier(nn.Module):
    def __init__(self, hidden_size=512):
        super(BgeClassifier, self).__init__()
        self.bge = AutoModel.from_pretrained(MODEL_CONFIGS['bge']['name'])
        self.bge.requires_grad_(False)  # 冻结 BGE 参数
        self.classifier = nn.Sequential(
            nn.Linear(self.bge.config.hidden_size, hidden_size),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)  # 256 -> 2（二分类）
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bge(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits

class QwenClassifier(nn.Module):
    def __init__(self, hidden_size=512):
        super(QwenClassifier, self).__init__()
        self.qwen = AutoModel.from_pretrained(MODEL_CONFIGS['qianwen']['name'])
        self.qwen.requires_grad_(False)  
        # 冻结 Qwen 参数
        for name, param in self.qwen.named_parameters():
            if 'layers.27' in name:
                param.requires_grad = True
        # 只解冻最后一层
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.qwen.config.hidden_size, 512),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 256 -> 2（二分类）
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 标记的隐藏状态（假设 Qwen 输出 last_hidden_state）
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 形状: [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)  # 应用 dropout
        logits = self.classifier(pooled_output)
        return logits



def get_model():
    model_class_name = MODEL_CONFIGS[CURRENT_MODEL]['class']
    if model_class_name == 'BertClassifier':
        return BertClassifier()
    elif model_class_name == 'BgeClassifier':
        return BgeClassifier()
    elif model_class_name == 'QwenClassifier':
        return QwenClassifier()
    else:
        raise ValueError(f"Unsupported model class: {model_class_name}")
    
def main():
    # 示例：加载模型并打印模型结构
    model = get_model()
    print(model)

if __name__ == "__main__":
    main()