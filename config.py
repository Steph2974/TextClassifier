# config.py
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data/Taiwan_text_classification_dateset.csv") 
MODEL_CHECKPOINT = "./bert" 
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "bert.pth")

# 训练超参数
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
DEVICE = 'mps'

# 数据拆分
TEST_SIZE = 0.2

CACHE_DIR = "siglip2/.cache"