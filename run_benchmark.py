#!/usr/bin/env python3
"""
ViktorAI Benchmark Runner

This script serves as a wrapper to run the benchmark script with the correct paths.
It ensures that all paths are resolved correctly regardless of the current working directory.
"""

import os
import sys
import argparse
import importlib.util

def main():
    """Main function to run the benchmark."""
    # Get the directory containing this script (the project root)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the path
    sys.path.insert(0, root_dir)
    
    # Import the benchmark module
    try:
        spec = importlib.util.spec_from_file_location("benchmark", os.path.join(root_dir, "tests", "benchmark.py"))
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
    except Exception as e:
        print(f"Error importing benchmark module: {e}")
        sys.exit(1)
    
    # Run the benchmark
    try:
        # Parse arguments directly from sys.argv, skipping the script name
        args = benchmark.parse_arguments()
        
        # Ensure character_data directory is found
        if not os.path.exists(os.path.join(root_dir, "character_data")):
            print("Warning: character_data directory not found in project root.")
        
        # Run the benchmark
        results = benchmark.main()
        print("Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 