import torch
from src.model import get_model
from config import *
from transformers import BertTokenizer, AutoTokenizer
import os

def load_tokenizer():
    tokenizer_class = MODEL_CONFIGS[CURRENT_MODEL]['tokenizer']
    model_name = MODEL_CONFIGS[CURRENT_MODEL]['name']
    if tokenizer_class == 'BertTokenizer':
        return BertTokenizer.from_pretrained(model_name)
    elif tokenizer_class == 'AutoTokenizer':
        return AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_class}")

def predict(text):
    # 加载tokenizer和模型
    tokenizer = load_tokenizer()
    model = get_model().to(DEVICE)
    model_path = os.path.join(MODEL_PATH, f'{CURRENT_MODEL}_model.pth')
    # 兼容保存的state_dict格式
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    # 文本预处理
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()

    label_map = {0: '异常', 1: '正常'}
    print(f"输入文本: {text}")
    print(f"预测类别: {label_map[pred]}")

if __name__ == '__main__':
    # text = input("请输入要分类的文本：")
    text = "想看看你的腿"
    predict(text)
    text = "今天天气不错，要出来约会吗？"
    predict(text)