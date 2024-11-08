import os
import logging
import random
import numpy as np
import torch
import argparse
from tqdm import trange, tqdm
from transformers import CLIPProcessor, CLIPImageProcessor
from transformers import AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from PIL import ImageFile
from models.mmsd20_model import MV_CLIPVisoBert
from data_utils.dataloader import Sarcasm
from data_utils.utils import collate_fn

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epochs')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in validation phase')
    parser.add_argument('--label_number', default=4, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--output_dir', default='./output_dir_CLIPVisoBert/', type=str, help='the output path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, device, train_loader, val_loader, processor_image, processor_text):
    os.makedirs(args.output_dir, exist_ok=True)
    model.to(device)
    
    optimizer = Adafactor(
        model.parameters(), lr=None, weight_decay=args.weight_decay,
        relative_step=True, scale_parameter=True, warmup_init=True
    )
    scheduler = AdafactorSchedule(optimizer)
    
    max_val_acc = 0
    output_model_dir = os.path.join(args.output_dir, "MV_CLIP")
    os.makedirs(output_model_dir, exist_ok=True)

    for i_epoch in trange(args.num_train_epochs, desc="Epoch"):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Iter", leave=False):
            text_list, image_list, label_list, _ = batch
            inputs_text = processor_text(text_list, return_tensors="pt", padding='max_length', truncation=True, max_length=args.text_size).to(device)
            inputs_image = processor_image(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)

            labels = torch.stack(label_list, dim=0).to(device)
            
            loss, _ = model(inputs_image=inputs_image, inputs_text=inputs_text, labels=labels)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch [{i_epoch+1}/{args.num_train_epochs}], Loss: {avg_loss:.4f}')

        val_f1, val_precision, val_recall = evaluate_f1(args, model, device, val_loader, processor_image, processor_text, mode='val')
        logger.info(f'Validation F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        
        # Save best model if improved
        if val_f1 > max_val_acc:
            max_val_acc = val_f1
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(output_model_dir, f"model_epoch_{i_epoch}.pt"))

    logger.info('Training complete')

def evaluate_f1(args, model, device, data_loader, processor_image, processor_text, mode='test'):
    model.eval()
    t_targets_all, t_outputs_all = [], []
    total_loss  = 0

    with torch.no_grad():
        for t_batch in data_loader:
            text_list, image_list, label_list, _ = t_batch
            inputs_text = processor_text(text_list, return_tensors="pt", padding='max_length', truncation=True, max_length=args.text_size).to(device)
            inputs_image = processor_image(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)

            labels = torch.stack(label_list, dim=0).to(device)

            loss, t_outputs = model(inputs_image, inputs_text, labels=labels)
            total_loss += loss.item()

            outputs = torch.argmax(t_outputs, -1)
            labels = torch.argmax(labels, -1)
            
            t_targets_all.extend(labels.detach().cpu().numpy())
            t_outputs_all.extend(outputs.detach().cpu().numpy())

    f1 = metrics.f1_score(t_targets_all, t_outputs_all, average='macro')
    precision = metrics.precision_score(t_targets_all, t_outputs_all, average='macro')
    recall = metrics.recall_score(t_targets_all, t_outputs_all, average='macro')

    return f1, precision, recall

def main():
    args = set_args()
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)

    # Load datasets
    train_set = Sarcasm(file_annotation="train/vimmsd_balanced_train.json", file_image="train/train-images")
    train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=args.seed)

    # Use DataLoader for batching
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.dev_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Initialize processor and model
    processor_image = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor_text = AutoTokenizer.from_pretrained("uitnlp/visobert")
    model = MV_CLIPVisoBert(args)
    #model = torch.nn.DataParallel(model) # For using multiple GPUs

    # Start training
    train(args, model, device, train_loader, val_loader, processor_image, processor_text)

if __name__ == "__main__":
    main()
