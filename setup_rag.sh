#!/bin/bash
# Setup script for CEE History RAG system

echo "Setting up CEE History RAG system..."
echo ""

# Install ChromaDB
echo "Installing ChromaDB..."
pip3 install --break-system-packages chromadb

# Create directories
echo "Creating directories..."
mkdir -p /home/data/documents
mkdir -p /home/data/chroma_db

# Test the setup
echo ""
echo "Testing RAG system..."
python3 /workspace/cee_rag.py

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your .txt documents in: /home/data/documents/"
echo "2. Run: python3 /workspace/cee_rag.py"
echo "3. Or use in your own scripts:"
echo "   from cee_rag import CEEHistoryRAG"
echo "   rag = CEEHistoryRAG()"
echo "   answer, sources = rag.query('your question')"
