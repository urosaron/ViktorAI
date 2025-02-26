"""
Indexer for ViktorAI.

This module processes and indexes Viktor's character data into a vector store
for efficient retrieval during conversations.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.config import Config
from src.vector_store import VectorStore

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk in characters.
        overlap: Overlap between chunks in characters.
        
    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the chunk size, save the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap from the previous chunk
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
        
        # Add the paragraph to the current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_sections_from_markdown(text: str) -> List[Tuple[str, str]]:
    """Extract sections from markdown text.
    
    Args:
        text: Markdown text.
        
    Returns:
        List of tuples (section_title, section_content).
    """
    # Split the text into sections based on headers
    sections = []
    current_section = ""
    current_content = []
    
    for line in text.split('\n'):
        if line.startswith('# ') or line.startswith('## '):
            # Save the previous section
            if current_section and current_content:
                sections.append((current_section, '\n'.join(current_content)))
            
            # Start a new section
            current_section = line
            current_content = []
        else:
            current_content.append(line)
    
    # Add the last section
    if current_section and current_content:
        sections.append((current_section, '\n'.join(current_content)))
    
    return sections

def process_character_data(config: Config) -> List[Dict]:
    """Process character data files into a list of documents with metadata.
    
    Args:
        config: Configuration object.
        
    Returns:
        List of dictionaries with text and metadata.
    """
    documents = []
    
    # Process core profile
    core_profile_path = config.get_character_file_path(config.core_profile_file)
    with open(core_profile_path, 'r', encoding='utf-8') as f:
        core_profile_text = f.read()
    
    sections = extract_sections_from_markdown(core_profile_text)
    for section_title, section_content in sections:
        documents.append({
            "text": f"{section_title}\n\n{section_content}",
            "metadata": {
                "source": "core_profile",
                "section": section_title.strip('# '),
                "type": "character_trait"
            }
        })
    
    # Process technical knowledge
    tech_knowledge_path = config.get_character_file_path(config.technical_knowledge_file)
    with open(tech_knowledge_path, 'r', encoding='utf-8') as f:
        tech_knowledge_text = f.read()
    
    sections = extract_sections_from_markdown(tech_knowledge_text)
    for section_title, section_content in sections:
        documents.append({
            "text": f"{section_title}\n\n{section_content}",
            "metadata": {
                "source": "technical_knowledge",
                "section": section_title.strip('# '),
                "type": "knowledge"
            }
        })
    
    # Process relationships
    relationships_path = config.get_character_file_path(config.relationships_file)
    with open(relationships_path, 'r', encoding='utf-8') as f:
        relationships_text = f.read()
    
    sections = extract_sections_from_markdown(relationships_text)
    for section_title, section_content in sections:
        documents.append({
            "text": f"{section_title}\n\n{section_content}",
            "metadata": {
                "source": "relationships",
                "section": section_title.strip('# '),
                "type": "relationship"
            }
        })
    
    # Process world context
    world_context_path = config.get_character_file_path(config.world_context_file)
    with open(world_context_path, 'r', encoding='utf-8') as f:
        world_context_text = f.read()
    
    sections = extract_sections_from_markdown(world_context_text)
    for section_title, section_content in sections:
        documents.append({
            "text": f"{section_title}\n\n{section_content}",
            "metadata": {
                "source": "world_context",
                "section": section_title.strip('# '),
                "type": "world_knowledge"
            }
        })
    
    # Process response guidelines
    guidelines_path = config.get_character_file_path(config.response_guidelines_file)
    with open(guidelines_path, 'r', encoding='utf-8') as f:
        guidelines_text = f.read()
    
    sections = extract_sections_from_markdown(guidelines_text)
    for section_title, section_content in sections:
        documents.append({
            "text": f"{section_title}\n\n{section_content}",
            "metadata": {
                "source": "response_guidelines",
                "section": section_title.strip('# '),
                "type": "guideline"
            }
        })
    
    # Process character analysis (this is a large file, so we process it differently)
    analysis_path = config.get_character_file_path(config.character_analysis_file)
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_text = f.read()
    
    sections = extract_sections_from_markdown(analysis_text)
    for section_title, section_content in sections:
        # For longer sections, split them into chunks
        if len(section_content) > 1000:
            chunks = split_text_into_chunks(section_content, chunk_size=500, overlap=100)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": f"{section_title} (Part {i+1}/{len(chunks)})\n\n{chunk}",
                    "metadata": {
                        "source": "character_analysis",
                        "section": section_title.strip('# '),
                        "part": i+1,
                        "total_parts": len(chunks),
                        "type": "scenes_and_events"
                    }
                })
        else:
            documents.append({
                "text": f"{section_title}\n\n{section_content}",
                "metadata": {
                    "source": "character_analysis",
                    "section": section_title.strip('# '),
                    "type": "scenes_and_events"
                }
            })
    
    return documents

def create_vector_store(config: Config, save_path: str = "vector_store") -> VectorStore:
    """Create and save a vector store from character data.
    
    Args:
        config: Configuration object.
        save_path: Path to save the vector store.
        
    Returns:
        The created vector store.
    """
    # Process character data
    documents = process_character_data(config)
    
    # Create vector store
    vector_store = VectorStore()
    
    # Add documents to vector store
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    vector_store.add_texts(texts, metadatas)
    
    # Save vector store
    vector_store.save_local(save_path)
    
    print(f"Created vector store with {len(documents)} documents and saved to {save_path}")
    
    return vector_store

def main():
    """Main function to create and save the vector store."""
    config = Config()
    create_vector_store(config)

if __name__ == "__main__":
    main() 