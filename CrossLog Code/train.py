import torch
from torch import nn
from torch.optim import lr_scheduler

from tqdm import tqdm

from dataloader import load_dataset
from model import Pipeline
from utils import set_random_seed, val, save_model


def train(args):
    set_random_seed()
    print(args)
    src_train_dataloader, src_val_dataloader = load_dataset(args, source=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Pipeline(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    bce_criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        print(f"Model training for epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        total_loss = 0.0
        anomaly_loss = 0.0
        sys_loss = 0.0
        mi_loss = 0.0
        for data in tqdm(src_train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            anomaly_logits, sys_logits, mi_loss = model(data)

            '''anomaly classification loss'''
            anomaly_classification_loss = bce_criterion(anomaly_logits, data.y_class.float())

            '''system classification loss'''
            sys_classification_loss = bce_criterion(sys_logits, data.y_sys.float())

            '''final loss (anomaly loss + system loss + MI loss)'''
            loss = (1 - args.alpha_sys - args.alpha_mi) * anomaly_classification_loss + args.alpha_sys * sys_classification_loss + args.alpha_mi * mi_loss

            loss.backward()
            optimizer.step()
            anomaly_loss += anomaly_classification_loss.item()
            sys_loss += sys_classification_loss.item()
            mi_loss += mi_loss.item()
            total_loss += loss.item()
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate for epoch {epoch + 1}: {current_lr}")
        scheduler.step()
        avg_anomaly_loss = anomaly_loss / len(src_train_dataloader)
        avg_mi_loss = mi_loss / len(src_train_dataloader)
        avg_sys_loss = sys_loss / len(src_train_dataloader)
        average_loss = total_loss / len(src_train_dataloader)  # avg loss
        print(
            f"training avg total loss {average_loss} | avg anomaly loss {avg_anomaly_loss} | avg system loss {avg_sys_loss} | avg mi loss {avg_mi_loss}\n")
        '''Validation'''
        f1, recall, precision, auprc, rocauc = val(model, src_val_dataloader, bce_criterion, args)
        '''Save model'''
        if f1 > best_f1:
            print(f'best f1 {f1}, save the model...')
            best_f1 = f1
            model_save_path = f'./models/towards_target_{args.target_dataset}_{args.dataset_id}_pretrain_checkpoint.pth'
            save_model(model, model_save_path)
