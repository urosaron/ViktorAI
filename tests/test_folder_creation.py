"""
Test script to verify folder creation functionality.
"""

from pathlib import Path
import os
import sys
from datetime import datetime

def test_folder_creation():
    """Test creating nested folders with timestamps."""
    try:
        # Base directory
        base_dir = Path("test_folder_creation")
        base_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created base directory: {base_dir}")
        
        # Model directory
        model_dir = base_dir / "test_model"
        model_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created model directory: {model_dir}")
        
        # Timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created run directory: {run_dir}")
        
        # Subdirectories
        raw_data_dir = run_dir / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        print(f"Created raw_data directory: {raw_data_dir}")
        
        vis_dir = run_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"Created visualizations directory: {vis_dir}")
        
        # Create a test file
        test_file = raw_data_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("Test file content")
        print(f"Created test file: {test_file}")
        
        print("\nFolder creation test successful!")
        return True
    except Exception as e:
        print(f"Error creating folders: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_folder_creation() 