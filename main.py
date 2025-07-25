# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import config  

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
model = AutoModelForMaskedLM.from_pretrained(config.MODEL_CHECKPOINT)