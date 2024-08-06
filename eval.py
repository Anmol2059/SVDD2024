import argparse
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import librosa
from Models.model_attention import Model
from utils import set_seed

def main(args):
    # Set the seed for reproducibility
    set_seed(args.random_seed)
    path = args.base_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the model
    print(f"Output will be on : {args.encoder}")
    model = Model(args, device=device).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {args.gpu_list} GPUs.")
        model = nn.DataParallel(model, device_ids=args.gpu_list)

    # Load the state dict of the saved model
    print(f"Loading model from: {args.model_path}")
    saved_state_dict = torch.load(args.model_path, map_location=device)

    # Get the state dict of the current model
    model_state_dict = model.state_dict()

    # Filter the saved state dict to only include keys that exist in the current model's state dict
    filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}

    # Update the current model's state dict with the filtered state dict
    model_state_dict.update(filtered_state_dict)

    # Load the updated state dict into the model
    model.load_state_dict(model_state_dict)
    print("Model loaded successfully.")

    model.eval()
    scores_out = args.output_path
    os.makedirs(scores_out, exist_ok=True)
    
    with torch.no_grad():
        for filename in tqdm(sorted(os.listdir(path)), desc=f"Testing"):
            audio_path = os.path.join(path, filename)
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            x = torch.tensor(audio).unsqueeze(0).to(device)
            pred = model(x)
            with open(os.path.join(scores_out, f'scores_{args.output_file_name}_randomseed_{args.random_seed}.txt'), "a") as f:
                f.write(f"{filename} {pred.item()}\n")
    # with torch.no_grad():
    #     for filename in tqdm(os.listdir(path), desc=f"Testing"):
    #         audio_path = os.path.join(path, filename)
    #         audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    #         x = torch.tensor(audio).unsqueeze(0).to(device)
    #         pred = model(x)
    #         with open(os.path.join(scores_out, f'scores_{args.output_file_name}_randomseed_{args.random_seed}.txt'), "a") as f:
    #             f.write(f"{filename} {pred.item()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model.")
    parser.add_argument("--gpu_list", type=int, nargs='+', default=[0], help="List of GPUs to use.")
    parser.add_argument("--output_file_name", type=str, default="tianchi-aug8-wavlmSE", help="Output file name.")
    parser.add_argument("--output_path", type=str, default="scores", help="The output folder for the scores.")
    args = parser.parse_args()
    main(args)