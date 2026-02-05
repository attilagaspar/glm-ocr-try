#!/usr/bin/env python3
"""
Simple RAG system for CEE Economic History documents
Uses Ollama (Qwen2.5:14b + nomic-embed-text) + ChromaDB
"""

import requests
import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings

class CEEHistoryRAG:
    """RAG system for historical documents"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.llm_model = "qwen2.5:14b"
        self.embed_model = "nomic-embed-text"
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory="/home/data/chroma_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="cee_history",
            metadata={"description": "CEE Economic History Documents"}
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using nomic-embed-text"""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.embed_model,
                "prompt": text
            }
        )
        return response.json()["embedding"]
    
    def add_document(self, text: str, metadata: Dict = None, doc_id: str = None):
        """Add a document to the vector database"""
        if doc_id is None:
            doc_id = f"doc_{self.collection.count()}"
        
        embedding = self.embed_text(text)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        print(f"Added document: {doc_id}")
    
    def add_documents_from_folder(self, folder_path: str, extension: str = ".txt"):
        """Add all documents from a folder"""
        folder = Path(folder_path)
        for file_path in folder.glob(f"*{extension}"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                metadata = {
                    "filename": file_path.name,
                    "source": str(file_path)
                }
                self.add_document(text, metadata, doc_id=file_path.stem)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def add_document_chunked(self, text: str, metadata: Dict = None, doc_id: str = None):
        """Add a document in chunks (better for large docs)"""
        chunks = self.chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}" if doc_id else f"chunk_{self.collection.count()}"
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            self.add_document(chunk, chunk_metadata, chunk_id)
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents"""
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate response using Qwen2.5"""
        if context:
            full_prompt = f"""Context from historical documents:
{context}

Question: {prompt}

Please answer based on the context provided above."""
        else:
            full_prompt = prompt
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": full_prompt,
                "stream": False
            }
        )
        
        return response.json()["response"]
    
    def query(self, question: str, n_results: int = 3) -> str:
        """Query the RAG system"""
        # Search for relevant context
        results = self.search(question, n_results)
        
        # Combine retrieved documents
        context = "\n\n---\n\n".join(results["documents"][0])
        
        # Generate response
        answer = self.generate_response(question, context)
        
        return answer, results
    
    def match_firm_names(self, firm_name: str, n_results: int = 10) -> List[Dict]:
        """Find similar firm names (fuzzy matching via embeddings)"""
        results = self.search(firm_name, n_results)
        
        matches = []
        for i, doc in enumerate(results["documents"][0]):
            matches.append({
                "text": doc,
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return matches


def main():
    """Example usage"""
    print("CEE History RAG System")
    print("=" * 60)
    
    # Initialize
    rag = CEEHistoryRAG()
    
    # Example 1: Add some sample documents
    print("\n1. Adding sample documents...")
    
    rag.add_document(
        "The Magyar Általános Kőszénbánya Részvénytársaság (Hungarian General Coal Mining Company) was established in 1891.",
        metadata={"type": "company", "year": 1891, "country": "Hungary"},
        doc_id="magyar_coal_1891"
    )
    
    rag.add_document(
        "In 1920, the Škoda Works employed over 35,000 workers, making it one of the largest industrial enterprises in Czechoslovakia.",
        metadata={"type": "company", "year": 1920, "country": "Czechoslovakia"},
        doc_id="skoda_1920"
    )
    
    rag.add_document(
        "The Romanian oil industry expanded rapidly between 1900-1914, with production increasing from 250,000 to 1.8 million tons.",
        metadata={"type": "industry", "period": "1900-1914", "country": "Romania"},
        doc_id="romania_oil"
    )
    
    # Example 2: Add documents from a folder
    print("\n2. To add all .txt files from a folder:")
    print("   rag.add_documents_from_folder('/home/data/documents')")
    
    # Example 3: Query the system
    print("\n3. Querying the system...")
    question = "Tell me about coal mining companies in Hungary"
    answer, results = rag.query(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\nSources used: {len(results['documents'][0])} documents")
    
    # Example 4: Firm name matching
    print("\n4. Finding similar firm names...")
    matches = rag.match_firm_names("Magyar Kőszénbánya")
    print(f"Found {len(matches)} similar entries")
    for match in matches[:3]:
        print(f"  - {match['text'][:100]}... (distance: {match['distance']:.3f})")
    
    print("\n" + "=" * 60)
    print("System ready! Use in your Python scripts:")
    print("  from cee_rag import CEEHistoryRAG")
    print("  rag = CEEHistoryRAG()")
    print("  answer, sources = rag.query('your question')")


if __name__ == "__main__":
    main()
