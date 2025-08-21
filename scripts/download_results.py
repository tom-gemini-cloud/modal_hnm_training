#!/usr/bin/env python3
"""
Script to download Modal training results locally for evaluation
"""
import os
import subprocess
import sys

def download_modal_results():
    """Download results from Modal volume to local directory"""
    
    # Get the directory where this script is located and go up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create local directory structure relative to project root
    local_dir = os.path.join(project_root, "data", "bert4rec_output", "bert4rec_20250821_102728")
    os.makedirs(local_dir, exist_ok=True)
    
    # Files to download
    files_to_download = [
        "results.json",
        "token_registry.json",
        "best_model.pt"
    ]
    
    for file_name in files_to_download:
        remote_path = f"bert4rec_output/bert4rec_20250821_102728/{file_name}"
        local_path = os.path.join(local_dir, file_name)
        
        print(f"Downloading {file_name}...")
        
        try:
            # Use modal volume get command
            result = subprocess.run([
                "modal", "volume", "get", "datasets", 
                remote_path, local_path
            ], capture_output=True, text=True, check=True)
            
            print(f"[OK] Successfully downloaded {file_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to download {file_name}: {e}")
            print(f"Error output: {e.stderr}")
            
        except Exception as e:
            print(f"[ERROR] Error downloading {file_name}: {e}")
    
    print(f"\nDownload complete! Results saved to: {local_dir}")
    
    # Verify downloads
    for file_name in files_to_download:
        file_path = os.path.join(local_dir, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"[OK] {file_name}: {size} bytes")
        else:
            print(f"[MISSING] {file_name}: Not found")

if __name__ == "__main__":
    download_modal_results()