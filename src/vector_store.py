"""
Vector Store for ViktorAI.

This module implements a vector database for storing and retrieving
Viktor's knowledge using sentence embeddings.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class SimpleVectorStore:
    """A simple vector store implementation using numpy arrays.
    
    This is a lightweight implementation that doesn't require external
    dependencies like FAISS. For production use with larger datasets,
    consider using a more robust solution.
    """
    
    def __init__(self, embedding_dimension: int = 384):
        """Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings.
        """
        self.embedding_dimension = embedding_dimension
        self.vectors = np.zeros((0, embedding_dimension), dtype=np.float32)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """Add documents and their embeddings to the store.
        
        Args:
            documents: List of document texts.
            embeddings: Matrix of embeddings with shape (n_documents, embedding_dimension).
            metadata: Optional list of metadata dictionaries for each document.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        if len(metadata) != len(documents):
            raise ValueError("Number of metadata items must match number of documents")
        
        # Add the new documents and embeddings
        self.vectors = np.vstack([self.vectors, embeddings])
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str, Dict]]:
        """Search for the most similar documents to the query.
        
        Args:
            query_embedding: Embedding of the query.
            top_k: Number of results to return.
            
        Returns:
            List of tuples (index, score, document, metadata).
        """
        if len(self.documents) == 0:
            return []
        
        # Normalize the query embedding and the document embeddings
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute cosine similarity
        scores = np.dot(self.vectors, query_embedding)
        
        # Get the indices of the top_k highest scores
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Return the results
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(scores[idx]),
                self.documents[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def save(self, file_path: str):
        """Save the vector store to disk.
        
        Args:
            file_path: Path to save the vector store.
        """
        data = {
            "embedding_dimension": self.embedding_dimension,
            "vectors": self.vectors.tolist(),
            "documents": self.documents,
            "metadata": self.metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'SimpleVectorStore':
        """Load a vector store from disk.
        
        Args:
            file_path: Path to the saved vector store.
            
        Returns:
            Loaded vector store.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        store = cls(embedding_dimension=data["embedding_dimension"])
        store.vectors = np.array(data["vectors"], dtype=np.float32)
        store.documents = data["documents"]
        store.metadata = data["metadata"]
        
        return store


class VectorStore:
    """Vector store for ViktorAI using sentence-transformers and FAISS if available."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        # Lazy import to avoid dependencies if not used
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False
            print("FAISS not available, falling back to simple vector store.")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("Sentence-transformers not available. Please install with: pip install sentence-transformers")
            self.model = None
            self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize the vector store
        if self.use_faiss and self.model is not None:
            import faiss
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents = []
            self.metadata = []
        else:
            self.simple_store = SimpleVectorStore(embedding_dimension=self.embedding_dimension)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: List of texts to add.
            metadatas: Optional list of metadata dictionaries for each text.
            
        Returns:
            List of IDs for the added texts.
        """
        if self.model is None:
            print("Sentence-transformers model not available. Cannot add texts.")
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Add to the appropriate store
        if self.use_faiss:
            self.index.add(embeddings.astype('float32'))
            start_idx = len(self.documents)
            self.documents.extend(texts)
            self.metadata.extend(metadatas)
            return [str(i) for i in range(start_idx, len(self.documents))]
        else:
            self.simple_store.add_documents(texts, embeddings, metadatas)
            return [str(i) for i in range(len(self.simple_store.documents) - len(texts), len(self.simple_store.documents))]
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar texts to the query.
        
        Args:
            query: Query text.
            k: Number of results to return.
            
        Returns:
            List of tuples (document, score).
        """
        if self.model is None:
            print("Sentence-transformers model not available. Cannot search.")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search in the appropriate store
        if self.use_faiss:
            import faiss
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # Ensure index is valid
                    results.append((self.documents[idx], float(scores[0][i])))
            return results
        else:
            search_results = self.simple_store.search(query_embedding, k)
            return [(doc, score) for _, score, doc, _ in search_results]
    
    def similarity_search_with_metadata(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar texts to the query and return with metadata.
        
        Args:
            query: Query text.
            k: Number of results to return.
            
        Returns:
            List of tuples (document, score, metadata).
        """
        if self.model is None:
            print("Sentence-transformers model not available. Cannot search.")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search in the appropriate store
        if self.use_faiss:
            import faiss
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # Ensure index is valid
                    results.append((
                        self.documents[idx], 
                        float(scores[0][i]),
                        self.metadata[idx]
                    ))
            return results
        else:
            search_results = self.simple_store.search(query_embedding, k)
            return [(doc, score, meta) for _, score, doc, meta in search_results]
    
    def save_local(self, folder_path: str):
        """Save the vector store to a local folder.
        
        Args:
            folder_path: Path to the folder to save the vector store.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        if self.use_faiss:
            import faiss
            import pickle
            
            # Save the FAISS index
            faiss.write_index(self.index, os.path.join(folder_path, "faiss_index.bin"))
            
            # Save the documents and metadata
            with open(os.path.join(folder_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
        else:
            self.simple_store.save(os.path.join(folder_path, "simple_store.json"))
    
    @classmethod
    def load_local(cls, folder_path: str, model_name: str = "all-MiniLM-L6-v2") -> 'VectorStore':
        """Load a vector store from a local folder.
        
        Args:
            folder_path: Path to the folder containing the saved vector store.
            model_name: Name of the sentence-transformers model to use.
            
        Returns:
            Loaded vector store.
        """
        store = cls(model_name=model_name)
        
        if os.path.exists(os.path.join(folder_path, "faiss_index.bin")):
            try:
                import faiss
                import pickle
                
                # Load the FAISS index
                store.index = faiss.read_index(os.path.join(folder_path, "faiss_index.bin"))
                
                # Load the documents and metadata
                with open(os.path.join(folder_path, "documents.pkl"), "rb") as f:
                    store.documents = pickle.load(f)
                
                with open(os.path.join(folder_path, "metadata.pkl"), "rb") as f:
                    store.metadata = pickle.load(f)
                
                store.use_faiss = True
                return store
            except ImportError:
                print("FAISS not available, falling back to simple vector store.")
        
        if os.path.exists(os.path.join(folder_path, "simple_store.json")):
            store.simple_store = SimpleVectorStore.load(os.path.join(folder_path, "simple_store.json"))
            store.use_faiss = False
            return store
        
        raise FileNotFoundError(f"No vector store found in {folder_path}")
    
    def query(self, query: str, top_k: int = 5, filter_fn=None) -> List[Tuple[str, float]]:
        """Search for similar texts to the query.
        
        This is an alias for similarity_search_with_score to maintain compatibility.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            filter_fn: Optional filter function to apply to results (ignored in this implementation).
            
        Returns:
            List of tuples (document, score).
        """
        # We ignore the filter_fn parameter since it's not needed for simple implementation
        return self.similarity_search_with_score(query, k=top_k) 