import os
import sys

def run_research_pipeline():
    scripts = [
        "download_dataset.py",
        "train.py", # Ensure this saves swin_dann_augmented_final.pth
        "advanced_model.py",
        "gradcam_metrics.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"\n>>> EXECUTING: {script}")
            exit_code = os.system(f"{sys.executable} {script}")
            if exit_code != 0:
                print(f"!!! Error in {script}. Pipeline halted.")
                break
        else:
            print(f"!!! Missing: {script}")

if __name__ == "__main__":
    run_research_pipeline()