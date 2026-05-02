import json
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from model.transformer import GPT
from model.utils import count_parameters

# Path where the summary should be saved (on the user's local workspace)
# Note: The user's error was in Colab, so they need this file on their Drive.
# I will create it locally first so they can see the content and then they can upload it,
# OR I can just give them the code to run in a Colab cell.

def create_summary():
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.safe_load(f)
    
    # Data from user's experiment log
    user_results = {
        "tiny": {"val": 1.1600, "time": 70, "tok_s": 1751104},
        "small": {"val": 0.9988, "time": 112, "tok_s": 1096438},
        "medium": {"val": 0.8560, "time": 201, "tok_s": 614248},
        "large": {"val": 0.7663, "time": 365, "tok_s": 337995},
        "xl": {"val": 0.6799, "time": 744, "tok_s": 165726}
    }
    
    summary = {}
    for name, cfg in model_configs.items():
        # Using default vocab_size=4096 as per config
        model = GPT.from_config_dict(cfg, vocab_size=4096)
        params = count_parameters(model, non_embedding=True)
        
        summary[name] = {
            "params": params,
            "final_val_loss": user_results[name]["val"],
            "best_val_loss": user_results[name]["val"], # same for 1 epoch
            "total_time_s": user_results[name]["time"],
            "tokens_per_sec": user_results[name]["tok_s"],
            "gpu_memory_gb": 0.0 # user didn't provide, but script needs it
        }
        print(f"Added {name}: {params:,} params, loss {user_results[name]['val']}")

    output_path = "sp_scaling_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCreated {output_path} successfully.")

if __name__ == "__main__":
    create_summary()
