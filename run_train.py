import os
import pprint
import argparse
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize
import torch.nn.functional as F

import clip
from model import CLIP, Bottleneck, AttentionPool2d
from simple_tokenizer import SimpleTokenizer


from train import train_main, load_data, load_clip, preprocess_text
from zero_shot import run_cxr_zero_shot, run_zero_shot, run_softmax_eval, make_true_labels
from eval import evaluate

import wandb
wandb.login()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/high_count_diseases_training.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/high_count_training_present_report.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--val_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/new_testing_without_lateral.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/high_count_diseases_training.csv', help="Directory to load labels from.")
    parser.add_argument('--val_label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/testing_data_without_lateral_only_nec.csv', help="Directory to load labels from.")
    parser.add_argument('--val_txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/testing_data_without_lateral_present_report.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--cxr_true_labels_path', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/testing_data_without_lateral_only_nec.csv', help="Directory to load true labels from.")
    parser.add_argument('--rare_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/low_count_diseases_training.h5', help="Directory to load rare labels from.")
    parser.add_argument('--rare_txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/low_count_training_present_report.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--rare_label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/low_count_diseases_training.csv', help="Directory to load labels from.")
    parser.add_argument('--bal_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/bal_low_count_diseases_training.h5', help="Directory to load rare labels from.")
    parser.add_argument('--bal_txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/bal_low_count_training_present_report.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--bal_label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/low_count_diseases_training_balanced.csv', help="Directory to load labels from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--best_model_dir', type=str, default="/home/vault/iwi5/iwi5207h/not_frozen/CheXzero/checkpoints/best_model/", help="Directory to save the best model.")
    parser.add_argument('--save_dir', type=str, default="/home/vault/iwi5/iwi5207h/not_frozen/CheXzero/checkpoints", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="vit_rare_base_min", help="Name of the model.")
    parser.add_argument('--plot_dir', type=str, default="/home/vault/iwi5/iwi5207h/not_frozen/CheXzero/plots/", help="Directory to save the plots.")
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help="Type of learning rate scheduler to use.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--factor', type=float, default=0.1, help="Factor by which the learning rate will be reduced.")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0):
    wandb.init(project="vit_rare_base_clip", config=config, name="min")
    
    # Make the model, data, and optimization problem
    model, train_loader,rare_loader,eval_loader, val_loader, device, c_criterion,b_criterion, optimizer, scheduler = make(config)

    # Train the model
    train(model, train_loader,rare_loader,eval_loader, val_loader, device, c_criterion,b_criterion, optimizer, scheduler, config, resume=True)

    # Save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    torch.save(model, model_path)
    
    wandb.save(model_path)
    wandb.finish()
    
    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = True
    train_loader, device = load_data(config.cxr_filepath, config.txt_filepath, config.label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    rare_loader, _ = load_data(config.rare_filepath, config.rare_txt_filepath,config.rare_label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    eval_loader, _ = load_data(config.val_filepath, config.val_txt_filepath,config.val_label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression",type = 'eval')
    val_loader, _ = load_data(config.val_filepath, config.val_txt_filepath, config.val_label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression", type = 'test')
    
    model = load_clip(model_path='/home/woody/iwi5/iwi5207h/current_trial/CheXzero/checkpoints/best_model/vit_chexzero_best_rare_3.pt', pretrained=pretrained, context_length=config.context_length)
    model.to(device)
    model = model.to(torch.float32)
    print('Model loaded and moved to device.')

    label_columns = ['Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 
                     'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 
                     'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 
                     'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 
                     'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 
                     'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
                     'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                     'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 
                     'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 
                     'Tortuous Aorta', 'Tuberculosis']
    class_weights = compute_class_weights(config.label_filepath, label_columns).to(device)


    # Define the criterion, optimizer, and scheduler
    c_criterion = nn.CrossEntropyLoss().to(device)
    b_criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0001)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    
    if config.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.patience, factor=config.factor)
    
    return model, train_loader,rare_loader, eval_loader, val_loader, device, c_criterion,b_criterion, optimizer, scheduler

def compute_class_weights(label_filepath, label_columns):
   
    labels_df = pd.read_csv(label_filepath)
    
    # Select only the label columns
    labels_df = labels_df[label_columns]
    
    # Sum occurrences of each class (assuming binary labels 0/1)
    class_counts = labels_df.sum(axis=0).values
    
    # Total number of samples
    total_samples = len(labels_df)
    
    # Handle cases where class counts might be zero by adding a small value (epsilon)
    epsilon = 1e-6
    class_counts = class_counts + epsilon
    
    # Compute class weights as inverse of frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Convert to torch tensor
    return torch.tensor(class_weights, dtype=torch.float32)

def freeze_vit_layers(model, strategy='gradual'):
    """
    Strategically freeze/unfreeze ViT and text transformer layers for fine-tuning on rare classes.
    
    Strategies:
    - 'gradual': Keep last layers and attention mechanisms trainable (default: last 4 layers)
    - 'attention_only': Only train attention mechanisms (attn heads + MLP)
    - 'minimal': Keep only final layer and projections trainable
    """
    
    def set_requires_grad(module, requires_grad=False, attention_only=False):
        if attention_only:
            for name, param in module.named_parameters():
                # Ensure that both attention heads and feedforward layers (MLP) are trainable in attention_only strategy
                param.requires_grad = requires_grad if 'attn' in name or 'mlp' in name else False
        else:
            for param in module.parameters():
                param.requires_grad = requires_grad

    # First freeze everything (ViT and text transformer)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreezing Vision Transformer (ViT) layers based on strategy
    if strategy == 'gradual':
        # Gradual strategy: Keep last 4 transformer blocks trainable
        num_layers = len(model.visual.transformer.resblocks)
        for i in range(num_layers - 4, num_layers):
            set_requires_grad(model.visual.transformer.resblocks[i], True)
        
        # Keep attention pooling trainable
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)
        
        # Keep final layer norm and projections trainable
        if hasattr(model.visual, 'ln_post'):
            set_requires_grad(model.visual.ln_post, True)
        if hasattr(model.visual, 'proj'):
            model.visual.proj.requires_grad = True
            
    elif strategy == 'attention_only':
        # Only train attention mechanisms + MLPs throughout the network
        for block in model.visual.transformer.resblocks:
            set_requires_grad(block, True, attention_only=True)
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)
            
    elif strategy == 'minimal':
        # Minimal strategy: Only train the final layer and projections
        if hasattr(model.visual, 'ln_post'):
            set_requires_grad(model.visual.ln_post, True)
        if hasattr(model.visual, 'proj'):
            model.visual.proj.requires_grad = True
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)

    # Unfreezing Text Transformer layers (similar to Vision Transformer)
    if strategy == 'gradual':
        num_text_layers = len(model.transformer.resblocks)
        for i in range(num_text_layers - 4, num_text_layers):
            set_requires_grad(model.transformer.resblocks[i], True)

    elif strategy == 'attention_only':
        # Only unfreeze attention heads + MLPs in text transformer
        for block in model.transformer.resblocks:
            set_requires_grad(block, True, attention_only=True)

    elif strategy == 'minimal':
        # Minimal strategy for text transformer: Keep final layers and projections trainable
        if hasattr(model, 'ln_final'):
            set_requires_grad(model.ln_final, True)

    # Always keep logit scale trainable for temperature scaling
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

    return model


def train(model, train_loader,rare_loader,eval_loader, val_loader, device, c_criterion,b_criterion, optimizer, scheduler, config, resume=False): 
    model.train()
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    wandb.watch(model, log="all")
    
    # Create the model save directory if it does not exist
    if not os.path.exists(model_save_dir): 
        os.makedirs(model_save_dir)

    batch_ct = 0
    start_epoch = 1
    report_freq = config.log_interval
    highest_val_auc = 0

    train_losses = []
    val_losses = []
    val_aucs = []
    val_loss = 0
    
    csv_file = os.path.join(model_save_dir, f'{config.model_name}_metrics.csv')
    eval_results, val_auc, val_loss = validate(model, eval_loader,val_loader, device, c_criterion,b_criterion, config.context_length, config.cxr_true_labels_path, config)
    # Resume training if necessary
    if resume and os.path.exists(model_save_dir):
        checkpoint_files = [file for file in os.listdir(model_save_dir) if file.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(model_save_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from checkpoint: {checkpoint_path} at epoch {start_epoch}")

            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        train_losses.append(float(row[1]))
                        val_losses.append(float(row[2]))
                        val_aucs.append(float(row[3]))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['optimizer']['param_groups'][0]['lr']

    # Training loop
    
    for epoch in range(start_epoch, config.epochs + 1):
        model= freeze_vit_layers(model, strategy='minimal')
        #train_loader = rare_loader
        '''if epoch >=5:
            model = freeze_vit_layers(model, strategy='gradual')
            train_loader = rare_loader
            config.save_interval = 30
            config.val_interval = 30
            config.log_interval = 30'''

        model.train()
        running_loss = 0.0

        for data in tqdm(train_loader, desc=f'Epoch {epoch}/{config.epochs}'):
            images = data['img']
            texts = data['txt']
            labels = data["label"]
            texts = preprocess_text(texts, model)
            total_loss = train_batch(images, texts,labels, model, device, c_criterion,b_criterion, optimizer)
            batch_ct += 1
            running_loss += total_loss.item()

            # Report and log training loss
            if (batch_ct % report_freq) == 0:
                avg_train_loss = running_loss / report_freq
                train_log(avg_train_loss, epoch)
                train_losses.append(avg_train_loss)
                wandb.log({"Train Loss": avg_train_loss})
                running_loss = 0.0
            
            # Save checkpoint
            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, f"checkpoint_{batch_ct}.pt")
                print(f"Saved checkpoint to: {model_path}")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                save(checkpoint, model_path)

                checkpoints = sorted([ckpt for ckpt in os.listdir(model_save_dir) if ckpt.startswith("checkpoint_")])
                if len(checkpoints) > 3:  # Keep only last 3 checkpoints
                    os.remove(os.path.join(model_save_dir, checkpoints[0]))

            # Validation
            if (batch_ct % config.val_interval) == 0:
                eval_results, val_auc, val_loss = validate(model, eval_loader,val_loader, device, c_criterion,b_criterion, config.context_length, config.cxr_true_labels_path, config)
                val_losses.append(val_loss)
                val_aucs.append(val_auc)

                # Save the best model
                if val_auc > highest_val_auc:
                    highest_val_auc = val_auc
                    save_best_model(model, optimizer, epoch, val_auc, config.best_model_dir)

                # Log validation metrics
                wandb.log({"Val AUC": val_auc, "Val Loss": val_loss})
                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_train_loss, val_loss, val_auc])

        if config.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
    
    # Plot metrics
    #plot_metrics(train_losses, val_losses, val_aucs, config)
    
    print('Training complete.')

def train_batch(images, texts,labels, model, device, c_criterion,b_criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    images, texts,labels= images.to(device), texts.to(device),labels.to(device)
    
    # Forward pass
    logits_per_image, logits_per_text,multi_label_logits = model(images, texts)
    #print(f'Logits per image: {logits_per_image}')
    #print(f'Logits per text: {logits_per_text}')

    batch_size = images.shape[0]
    c_labels = torch.arange(batch_size).to(device)
    # Compute loss
    loss_img = c_criterion(logits_per_image, c_labels)
    loss_txt = c_criterion(logits_per_text, c_labels)

    #multi_loss = b_criterion(multi_label_logits, labels)

    #contrastive_loss_value = (loss_img + loss_txt) / 2
    #print(f'Loss Image: {loss_img}')
    #print(f'Loss Text: {loss_txt}') 

    '''# Compute KL divergence between the image and text distributions
    kl_divergence_value = F.kl_div(image_distribution.log(), text_distribution, reduction='batchmean')
    alpha = 0.5
    beta = 0.5
    total_loss = alpha * contrastive_loss_value + beta * kl_divergence_value'''
    total_loss = (loss_img + loss_txt) / 2
    #total_loss += multi_loss
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    return total_loss

def validate(model, eval_loader,val_loader, device, c_criterion,b_criterion, context_length, cxr_true_labels_path, config):
    model.eval()
    total_loss = 0
    all_images = []
    all_texts = []
    pair_template = ("{}", "no {}")
    cxr_labels = [
        'Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 'Cardiomegaly',
        'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum',
        'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 'Hernia', 'Hydropneumothorax', 'Infarction',
        'Infiltration', 'Kyphosis', 'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule',
        'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia',
        'Pneumoperitoneum', 'Pneumothorax', 'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture',
        'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'Tuberculosis'
    ]
    model.to(device)
    model = model.to(torch.float32)
    with torch.no_grad():
        for batch in val_loader:
            images, texts,labels = batch['img'], batch['txt'],batch['label']
            #print(f'Images: {images}')
            #print(f'Texts: {texts}')
            # Ensure images are tensors before moving to device
            if isinstance(images, list):
                images = torch.stack(images)
        

            # Preprocess texts before stacking (tokenization or embedding)
            texts = preprocess_text(texts, model)
            images, texts,label = images.to(device), texts.to(device),labels.to(device)
            #print(f'Images: {images}')
           # print(f'Texts: {texts}')
            logits_per_image, logits_per_text,multi_label_logits = model(images, texts)
            #print(f'Logits per image: {logits_per_image}')
            #print(f'Logits per text: {logits_per_text}')    
            # Check for NaNs in logits

            batch_size = images.shape[0]
            contrastive_labels = torch.arange(batch_size).to(device)

            loss_img = c_criterion(logits_per_image, contrastive_labels)
            loss_txt = c_criterion(logits_per_text, contrastive_labels)

            '''contrastive_loss_value = (loss_img + loss_txt) / 2
            kl_divergence_value = F.kl_div(image_distribution.log(), text_distribution, reduction='batchmean')
            alpha = 0.5
            beta = 0.5
            loss = alpha * contrastive_loss_value + beta * kl_divergence_value'''
            loss = (loss_img + loss_txt) / 2
            labels = labels.to(multi_label_logits.device)
            #multi_loss = b_criterion(multi_label_logits, labels)
            #loss += multi_loss

            total_loss += loss.item()
        avg_val_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss}')

    # Get predictions and evaluate
    y_pred = run_softmax_eval(model, eval_loader, cxr_labels, pair_template, context_length)
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
    # Ensure test_true is the same size as y_pred
    test_true = test_true[:len(y_pred)]
    eval_results = evaluate(y_pred, test_true, cxr_labels)
    
    # Log evaluation results for each label
    # Convert eval_results to a dictionary
    eval_results_dict = eval_results.to_dict(orient='list')

    # Log evaluation results for each label
    for label in cxr_labels:
        auc_key = f"{label}_auc"
        if auc_key in eval_results_dict:
            # Log the mean AUC value for the label
            auc_value = eval_results_dict[auc_key][0]
            wandb.log({auc_key: auc_value})
    
    val_auc = eval_results.filter(like='_auc').mean(axis=1).mean()
    
    
    return eval_results, val_auc, avg_val_loss

def plot_metrics(train_losses, val_losses, val_aucs, config):
    # Plot Train Loss, Val Loss, and Val AUC
    plt.figure(figsize=(10, 5))
    
    # Plot train loss
    plt.plot(train_losses, label='Train Loss', color='blue')
    
    # Plot val loss
    if val_losses:
        plt.plot(val_losses, label='Val Loss', color='red')
    
    # Plot val AUC
    if val_aucs:
        plt.plot(val_aucs, label='Val AUC', color='green')

    plt.xlabel('Batch Steps')
    plt.ylabel('Metrics')
    plt.title(f'{config.model_name} Training/Validation Metrics')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(config.plot_dir, f'{config.model_name}_metrics_plot.png')
    plt.savefig(plot_path)
    plt.show()

def save(checkpoint, path):
    torch.save(checkpoint, path)

def save_best_model(model, optimizer, epoch, val_auc, model_save_dir):
    best_model_path = os.path.join(model_save_dir, "vit_rare_base_clip_min.pt")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_auc': val_auc,
    }
    save(checkpoint, best_model_path)
    print(f"Best model saved at {best_model_path}")

def train_log(avg_train_loss, epoch):
    print(f'Epoch {epoch}: Train Loss = {avg_train_loss}')

if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
