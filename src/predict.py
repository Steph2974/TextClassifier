import torch
from transformers import BertTokenizer, AutoTokenizer
from src.model import get_model
from config import *
import os

def predict(texts):
    model = get_model()
    # model_path = os.path.join(MODEL_PATH, f'{CURRENT_MODEL}_epoch5.pth') # ✅
    model_path = '/Users/liang/Desktop/TextClassifier/models-bert-v2/bert_final.pth'
    print(f"Loading model from {model_path}") # ✅
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    
    tokenizer_class = MODEL_CONFIGS[CURRENT_MODEL]['tokenizer']
    model_name = MODEL_CONFIGS[CURRENT_MODEL]['name']
    if tokenizer_class == 'BertTokenizer':
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif tokenizer_class == 'AutoTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_class}")
    
    encodings = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"Outputs: {outputs}") # ✅
        predictions = torch.argmax(outputs, dim=1)
        print(f"Predictions: {predictions}")  # Debugging line to check predictions
    
    return predictions.cpu().numpy()

if __name__ == '__main__':
    # 示例文本
    sample_texts = [
        "想看看你的腿",
        "今天天气不错，要出来约会吗？"
    ]
    predictions = predict(sample_texts)
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {'正常' if pred == 1 else '色情'}")