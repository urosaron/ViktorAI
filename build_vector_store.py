#!/usr/bin/env python3
"""
Build Vector Store for ViktorAI.

This script builds a vector store from Viktor's character data for use in RAG.
"""

import os
import sys
from src.config import Config
from src.indexer import create_vector_store

def main():
    """Main function to build the vector store."""
    print("Building vector store for ViktorAI...")
    
    # Initialize configuration
    config = Config()
    
    # Create vector store
    vector_store = create_vector_store(config)
    
    print("Vector store built successfully.")
    print("You can now run ViktorAI with RAG capabilities.")

if __name__ == "__main__":
    sys.exit(main() or 0) 