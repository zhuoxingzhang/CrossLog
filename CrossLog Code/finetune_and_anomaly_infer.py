import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataloader import load_dataset
from model import Pipeline
from utils import show_metrics, set_random_seed, load_model, find_best_threshold, val, \
    save_model


def binary_accuracy_from_probs(pred_probs, true_labels, threshold=0.5):
    correct = 0
    for p, y in zip(pred_probs, true_labels):
        pred = 1 if p >= threshold else 0
        if pred == y:
            correct += 1
    return correct / len(true_labels)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def reinit_module(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def finetune(model, train_dataloader, val_dataloader, test_dataloader, args):
    '''freeze parameters for system-agnostic modules'''
    freeze_module(model.sys_agnostic_gnn)
    freeze_module(model.sys_agnostic_mlp)
    freeze_module(model.sys_agnostic_gated_subgraph_net)
    model.domain_invar_layer.requires_grad = False
    freeze_module(model.multihead_attn)
    freeze_module(model.multihead_attn_semantic)
    model.theta_attn.requires_grad = False
    freeze_module(model.sys_classifier)
    '''initialize parameters for system-specific mudules'''
    model.sys_specific_gnn.apply(reinit_module)
    model.sys_specific_mlp.apply(reinit_module)
    model.anomaly_classifier.apply(reinit_module)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ft_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.ft_step_size, gamma=args.ft_gamma)

    bce_criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_f1 = 0.0
    for epoch in range(args.ft_num_epochs):
        print(f"Model finetuning for epoch {epoch + 1}/{args.ft_num_epochs}")
        model.train()
        total_loss = 0.0
        for data in tqdm(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            anomaly_logits, _, _ = model(data)
            '''anomaly classification loss'''
            anomaly_classification_loss = bce_criterion(anomaly_logits, data.y_class.float())

            anomaly_classification_loss.backward()
            optimizer.step()
            total_loss += anomaly_classification_loss.item()
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate for epoch {epoch + 1}: {current_lr}")
        scheduler.step()
        average_loss = total_loss / len(train_dataloader)  # avg loss
        print(f"finetuning average loss {average_loss}\n")
        '''Validation'''
        f1, recall, precision, auprc, rocauc = val(model, val_dataloader, bce_criterion, args)
        '''Save model'''
        if f1 > best_f1:
            print(f'best f1 {f1}, save the model...')
            best_f1 = f1
            model_save_path = f'./models/towards_target_{args.target_dataset}_{args.dataset_id}_{args.ratio_finetuning}_{args.num_finetuning}_checkpoint.pth'
            save_model(model, model_save_path)


def infer(model, test_dataloader, args):
    model.eval()
    test_true_class_labels = []
    test_predicted_class_prob = []
    total_infer_time = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = batch.to(args.device)

            start_infer_time = time.time()
            anomaly_logits, _, _ = model(batch)
            y_class_prob = torch.sigmoid(anomaly_logits)
            end_infer_time = time.time()
            total_infer_time += (end_infer_time - start_infer_time)

            test_predicted_class_prob.extend(y_class_prob.cpu().numpy())
            test_true_class_labels.extend(batch.y_class.cpu().numpy())

    best_f1, best_recall, best_precision, best_threshold = find_best_threshold(test_true_class_labels,
                                                                               test_predicted_class_prob, 0, 1.0, 0.01)
    test_predicted_class_labels = (np.array(test_predicted_class_prob) >= best_threshold).astype(int)
    f1, recall, precision, auprc, rocauc = show_metrics(test_true_class_labels, test_predicted_class_labels,
                                                        test_predicted_class_prob)
    perf = f"Test anomaly classification performance (f1, recall, precision, prauc, rocauc) | {best_f1}, {best_recall}, {best_precision}, {auprc}, {rocauc}"
    cost = f"Inference time: {(total_infer_time / 3600):.4f} hours"
    print(perf)
    print(cost)
    os.makedirs("./log", exist_ok=True)
    log_file = f"./log/test_results_{args.dataset_id}.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(str(args) + "\n")
        f.write(perf + "\n")
        f.write(cost + "\n")
        f.write("#" * 80 + "\n\n")


def finetune_and_infer(args):
    set_random_seed()
    print(args)
    target_train_dataloader, target_val_dataloader, target_test_dataloader = load_dataset(args, source=False)
    """LOAD MODEL"""
    model = Pipeline(args).to(args.device)
    model_save_path = f'./models/towards_target_{args.target_dataset}_{args.dataset_id}_pretrain_checkpoint.pth'
    model = load_model(model, model_save_path)
    if args.num_finetuning != 0:  # not for zero-shot
        print("Start to finetune models...")
        finetune(model, target_train_dataloader, target_val_dataloader, target_test_dataloader, args)
        """LOAD FINE-TUNED MODEL"""
        model = Pipeline(args).to(args.device)
        model_save_path = f'./models/towards_target_{args.target_dataset}_{args.dataset_id}_{args.ratio_finetuning}_{args.num_finetuning}_checkpoint.pth'
        model = load_model(model, model_save_path)
    print("\nStart to infer anomalies...")
    infer(model, target_test_dataloader, args)
