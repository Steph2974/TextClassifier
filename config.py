import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #TODO：怎么知道BASE_DIR是正确的？
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Taiwan_text_classification_dateset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models') # 用于保存模型
LOG_DIR = os.path.join(BASE_DIR, 'logs') # 用于保存日志文件

# 模型配置，可以通过模型名获取模型相关信息
MODEL_CONFIGS = {
    'bert': {
        'name': "google-bert/bert-base-chinese",
        'class': 'BertClassifier',
        'tokenizer': 'BertTokenizer'
    },
    'bge': {
        'name': 'BAAI/bge-large-zh-v1.5',
        'class': 'BgeClassifier',
        'tokenizer': 'AutoTokenizer'
    },
    'qianwen': {
        'name': "Qwen/Qwen3-Embedding-0.6B", # 'name': "Qwen/Qwen3-4B-Base"'Qwen/Qwen2.5-7B-Instruct'
        'class': 'QwenClassifier',
        'tokenizer': 'AutoTokenizer'
    }
}

# 当前使用的模型
CURRENT_MODEL = 'qianwen'  # 可切换为 'bge' 或 'qianwen'

# 通用配置
NUM_CLASSES = 2
MAX_LENGTH = 64
BATCH_SIZE =8 # 增大了batch_size，bert 是 32，bge 是 8，qwen 是 8
EPOCHS = 15
LEARNING_RATE = 1e-2
DEVICE = 'mps'

# 创建目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)