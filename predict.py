import os
import json
import torch
import argparse
from transformers import CLIPProcessor
from models.mmsd20_model import MV_CLIP
from data_utils.dataloader import Sarcasm
from torch.utils.data import DataLoader
from data_utils.utils import collate_fn
from tqdm import tqdm
from PIL import ImageFile

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_MAPPING = {0: 'not-sarcasm', 1: 'multi-sarcasm', 2: 'image-sarcasm', 3: 'text-sarcasm'}

def set_args():
    root_output = '../output_dir/MV_CLIP'
    L_model_state = sorted(os.listdir(root_output))
    model_path = os.path.join(root_output, L_model_state[-1])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=model_path, type=str, help='path to the saved model checkpoint')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for predictions')
    parser.add_argument('--output_file', default='results.json', type=str, help='file to save predictions')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument('--label_number', default=4, type=int, help='the number of classification labels')
    return parser.parse_args()

def load_model(args, device):
    # Load the trained model
    model = MV_CLIP(args)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_result(predictions, phase):
    results = {id: label for id, label in enumerate(predictions)}
    result_final = {
        "results": results,
        "phase": phase
    }
    
    return result_final

def predict(args, model, data_loader, processor, device):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            text_list, image_list, _, _ = batch  # Only text and image data needed for inference

            inputs = processor(
                text=text_list,
                images=image_list,
                padding="max_length",
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt"
            ).to(device)
            
            # Perform inference
            outputs = model(inputs, labels=None)
            predicted_labels = torch.argmax(outputs[0], -1).cpu().numpy()
            predictions.extend([LABEL_MAPPING[label] for label in predicted_labels])

    return predictions

def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and processor
    model = load_model(args, device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the test dataset
    test_set = Sarcasm(file_annotation="public_test/vimmsd_public_test.json", file_image="public_test/public-test-images")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Generate predictions
    predictions = predict(args, model, test_loader, processor, device)
    
    result_final = create_result(predictions, 'dev')

    # Save predictions to file
    with open(args.output_file, 'w') as f:
        json.dump(result_final, f)

    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()
