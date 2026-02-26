import subprocess
import os
import glob
import json
import pandas as pd
import sys

def main():
    models = ["conv_vae", "fc_vae", "aae", "beta_vae"]
    # Check available datasets
    dataset_files = glob.glob("datasets/IAD/*_Train_Test.pkl")
    suffix = "_Train_Test.pkl"
    datasets = []
    for f in dataset_files:
        filename = os.path.basename(f)
        if filename.endswith(suffix):
            datasets.append(filename[:-len(suffix)])
    
    print(f"Found datasets: {datasets}")
    
    # Configuration
    epochs = 5 # Lower epochs for demonstration/testing, increase for real results
    
    for dataset in datasets:
        for model in models:
            print(f"==================================================")
            print(f"Checking {model} on {dataset}")
            
            # Check if done
            run_name_arg = "experiment_1"
            out_dir = os.path.join("runs", f"{dataset}_{model}_{run_name_arg}")
            if os.path.exists(os.path.join(out_dir, "best_stats.json")):
                print(f"Skipping {model} on {dataset}, already done.")
                continue

            print(f"Running {model} on {dataset}")
            print(f"==================================================")
            
            cmd = [
                sys.executable, "src/train.py",
                "--dataset", dataset,
                "--model", model,
                "--epochs", str(epochs),
                "--run", run_name_arg
            ]
            
            # For beta-VAE, we might want to try different betas, but default is 1.0 (same as VAE)
            if model == "beta_vae":
                cmd.extend(["--beta", "4.0"]) 
                
            subprocess.run(cmd, check=True)

    # Aggregate Results
    print("\nAggregating Results...")
    results = []
    
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        for run_folder in os.listdir(runs_dir):
            stats_path = os.path.join(runs_dir, run_folder, "best_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                
                # Parse folder name: dataset_model_runname
                parts = run_folder.split("_")
                # Warning: Splitting by underscore might be fragile if dataset name contains underscore
                # Assuming "dataset_model_runname" structure and dataset/model are known.
                # However, dataset names like "16QAM" are fine. "conv_vae" has underscore.
                # Let's try to match known models.
                
                dataset_name = "unknown"
                model_name = "unknown"
                
                # Heuristic parsing
                for m in models:
                    if m in run_folder:
                        model_name = m
                        break
                
                for d in datasets:
                    if run_folder.startswith(d):
                        dataset_name = d
                        break
                        
                entry = {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "MAE_AUC": stats.get("mae_auc"),
                    "MSE_AUC": stats.get("mse_auc"),
                    "PER_AUC": stats.get("per_auc"),
                    "Rel_AUC": stats.get("rel_auc"),
                    "Epoch": stats.get("epoch")
                }
                results.append(entry)

    if results:
        df = pd.DataFrame(results)
        print("\nResults Summary:")
        print(df)
        df.to_csv("experiment_results.csv", index=False)
        print("Saved to experiment_results.csv")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
