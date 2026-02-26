import subprocess
import os
import sys

def main():
    dataset = "16QAM"
    epochs = 50
    run_name = "tuning_exp_aggressive"
    
    # Configurations to test
    configs = [
        # (d_hidden, recon_weight, adv_weight)
        (512, 0.001, 1.0),   # Balance ~50 vs 1
        (512, 0.0001, 1.0),  # Balance ~5 vs 1
        (512, 0.00001, 1.0), # Balance ~0.5 vs 1
    ]
    
    print(f"Starting AAE Tuning on {dataset} for {epochs} epochs...")
    
    for (dh, rw, aw) in configs:
        print(f"==================================================")
        print(f"Testing Config: d_hidden={dh}, rw={rw}, aw={aw}")
        print(f"==================================================")
        
        cmd = [
            sys.executable, "src/train_aae_v2.py",
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--run", run_name,
            "--d_hidden", str(dh),
            "--recon_weight", str(rw),
            "--adv_weight", str(aw)
        ]
        
        subprocess.run(cmd, check=True)

    print("All tuning experiments completed.")

if __name__ == "__main__":
    main()
