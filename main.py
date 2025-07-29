# main.py
import argparse
from src.train import train
from src.predict import predict
from config import *
import torch
from transformers import AutoProcessor
import os

def main():
    parser = argparse.ArgumentParser(description="Face Classification with SigLIP 2")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode: train or infer")
    parser.add_argument("--text", type=str, help="Text for inference")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "infer":
        predictions = predict([args.text])
        print(f"Prediction for '{args.text}': {predictions[0]}")

if __name__ == "__main__":
    main()

# python main.py --mode infer --text "想看看你的腿"
# python main.py --mode infer --text "想摸摸你的腿"