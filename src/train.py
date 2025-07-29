import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.dataset import TextDataset
from src.model import get_model
from config import *
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
from tqdm import tqdm

# 设置日志
logging.basicConfig(filename=os.path.join(LOG_DIR, f'train_{CURRENT_MODEL}.log'), level=logging.INFO)

def log_trainable_params(model, epoch, prefix=""):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{prefix}Epoch {epoch+1} 可训练参数: {name}, 均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}")

def train():
    # 加载数据
    dataset = TextDataset(DATA_PATH)
    length = len(dataset)
    logging.info(f"数据集长度: {length}")

    train_size = int(0.7 * length)
    val_size = int(0.15 * length)
    test_size = length - train_size - val_size
    logging.info(f"训练集长度: {train_size}, 验证集长度: {val_size}, 测试集长度: {test_size}")

    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    # 训练集打乱，验证集和测试集不打乱
    # num_workers=4 是什么意思？使用 4 个子进程来并行加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 初始化模型和优化器
    model = get_model().to(DEVICE)
    logging.info(f'模型初始化完成')
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    logging.info(f'优化器初始化完成')
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) # 学习率衰减
    
    # 训练循环
    logging.info(f'开始训练')
    logging.info(f'训练轮数: {EPOCHS}')
    logging.info(f'批次大小: {BATCH_SIZE}')
    logging.info(f'训练批次: {len(train_dataloader)}')
    logging.info(f'学习率: {LEARNING_RATE}')
    logging.info(f'优化器: {optimizer}')
    logging.info(f'学习率衰减: {scheduler}') 
    
    for epoch in range(EPOCHS):
        # 记录epoch开始时可训练参数的均值和标准差
        log_trainable_params(model, epoch, prefix="[开始] ")
        logging.info(f'Epoch {epoch+1}/{EPOCHS}')
        model.train() # 训练模式
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', 
                   leave=True, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # 在数据集类的__getitem__方法中，input_ids和attention_mask是张量，labels是整数
            input_ids = batch['input_ids'].to(DEVICE) 
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            # logging.info(f'outputs: {outputs}')
            # logging.info(f'labels: {labels}')
            
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            # logging.info(f'loss: {loss}')
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 更新进度条信息
            current_loss = loss.item()
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Avg_Loss': f'{avg_loss_so_far:.4f}',
                'Batch': f'{batch_idx+1}/{len(train_dataloader)}'
            })
        
        # 记录epoch结束时可训练参数的均值和标准差
        log_trainable_params(model, epoch, prefix="[结束] ")
        avg_loss = total_loss / len(train_dataloader)
        logging.info(f'Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}')

        # 验证阶段
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_dataloader, DEVICE)
        
        # 记录训练信息
        logging.info(f'Epoch {epoch+1}/{EPOCHS}')
        logging.info(f'  训练损失: {avg_loss:.4f}')
        logging.info(f'  验证损失: {val_loss:.4f}')
        logging.info(f'  验证准确率: {val_accuracy:.4f}')

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            model_path = os.path.join(MODEL_PATH, f'{CURRENT_MODEL}_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f'模型已保存到: {model_path}')

    # 测试阶段
    test_loss, test_accuracy, test_predictions, test_labels = evaluate_model(model, test_dataloader, DEVICE)
     
    # 生成详细测试报告
    report = classification_report(test_labels, test_predictions, 
                                target_names=['异常', '正常'], 
                                output_dict=True)
    
    logging.info(f'最终测试结果:')
    logging.info(f'  测试损失: {test_loss:.4f}')
    logging.info(f'  测试准确率: {test_accuracy:.4f}')
    logging.info(f'  详细分类报告:')
    logging.info(f'    异常类 - 精确率: {report["异常"]["precision"]:.4f}, 召回率: {report["异常"]["recall"]:.4f}, F1: {report["异常"]["f1-score"]:.4f}')
    logging.info(f'    正常类 - 精确率: {report["正常"]["precision"]:.4f}, 召回率: {report["正常"]["recall"]:.4f}, F1: {report["正常"]["f1-score"]:.4f}')

    # 保存最终模型
    final_model_path = os.path.join(MODEL_PATH, f'{CURRENT_MODEL}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'最终模型已保存到: {final_model_path}')

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # 为验证/测试也添加进度条
    pbar = tqdm(dataloader, desc='Evaluating', leave=False, ncols=100)
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

if __name__ == '__main__':
    train()