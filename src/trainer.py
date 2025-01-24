import torch
import logging
from .visualize import visualize_predictions

def train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, num_epochs=20, save_dir='./predictions'):
    logging.basicConfig(filename='logs/training_logs.log', level=logging.INFO)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits_mask = model(images)
            loss = model.compute_loss(logits_mask, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        scheduler.step()
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss / len(train_dataloader)}")

        # 验证过程
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_dataloader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                logits_mask = model(images)
                loss = model.compute_loss(logits_mask, masks)
                valid_loss += loss.item()

            logging.info(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {valid_loss / len(valid_dataloader)}")

        # 每个epoch保存预测图像
        visualize_predictions(images.cpu().numpy(), masks.cpu().numpy(), logits_mask.sigmoid().cpu().numpy(), save_dir)
