import subprocess
import os
import sys

def main():
    models = ["conv_vae", "fc_vae", "aae", "beta_vae"]
    dataset = "16QAM"
    epochs = 200
    run_name = "final_200ep"
    
    print(f"Starting Final Experiment on {dataset} for {epochs} epochs...")
    
    for model in models:
        print(f"==================================================")
        print(f"Running {model} on {dataset}")
        print(f"==================================================")
        
        cmd = [
            sys.executable, "src/train.py",
            "--dataset", dataset,
            "--model", model,
            "--epochs", str(epochs),
            "--run", run_name
        ]
        
        if model == "beta_vae":
            cmd.extend(["--beta", "4.0"]) 
            
        subprocess.run(cmd, check=True)

    print("All experiments completed.")

if __name__ == "__main__":
    main()
