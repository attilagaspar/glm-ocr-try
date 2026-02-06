#!/usr/bin/env python3
"""
Interactive chat interface for CEE History RAG system
"""

import sys
from cee_rag import CEEHistoryRAG

def print_separator():
    """Print a visual separator"""
    print("\n" + "─" * 80 + "\n")

def print_sources(results):
    """Print source information from results"""
    print("\n📚 Sources used:")
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        filename = metadata.get("filename", "unknown")
        chunk_info = ""
        if "chunk_index" in metadata:
            chunk_info = f" (chunk {metadata['chunk_index']+1}/{metadata['total_chunks']})"
        print(f"  [{i}] {filename}{chunk_info}")
        # Show first 150 chars of the source
        preview = doc[:150].replace("\n", " ")
        if len(doc) > 150:
            preview += "..."
        print(f"      {preview}")

def main():
    """Run interactive chat"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CEE Economic History RAG Chat" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nInitializing RAG system...")
    try:
        rag = CEEHistoryRAG()
    except Exception as e:
        print(f"❌ ERROR: Could not initialize RAG system: {e}")
        print("\nMake sure:")
        print("  1. Docker container is running")
        print("  2. Ollama server is running: ollama serve > /dev/null 2>&1 &")
        print("  3. Models are pulled: ollama pull qwen2.5:14b && ollama pull nomic-embed-text")
        sys.exit(1)
    
    print("✓ RAG system ready!")
    
    # Show collection stats
    doc_count = rag.collection.count()
    print(f"\n📊 Database stats:")
    print(f"   Documents in collection: {doc_count}")
    
    if doc_count == 0:
        print("\n⚠️  WARNING: No documents in the collection!")
        print("   Add documents first using cee_rag.py or add_documents_from_folder()")
    
    print("\nCommands:")
    print("  Type your question and press Enter")
    print("  'quit' or 'exit' to exit")
    print("  'sources on/off' to toggle source display")
    print("  'n <number>' to change number of sources retrieved (default: 3)")
    print("  'stats' to show database statistics")
    
    # Settings
    show_sources = True
    n_results = 3
    
    while True:
        print_separator()
        
        # Get user input
        try:
            question = input("🔍 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        # Handle empty input
        if not question:
            continue
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if question.lower() == 'sources on':
            show_sources = True
            print("✓ Source display enabled")
            continue
        
        if question.lower() == 'sources off':
            show_sources = False
            print("✓ Source display disabled")
            continue
        
        if question.lower().startswith('n '):
            try:
                n_results = int(question.split()[1])
                print(f"✓ Number of sources set to {n_results}")
            except:
                print("❌ Invalid format. Use: n <number>")
            continue
        
        if question.lower() == 'stats':
            doc_count = rag.collection.count()
            print(f"\n📊 Database statistics:")
            print(f"   Total documents: {doc_count}")
            print(f"   Collection name: {rag.collection.name}")
            print(f"   Persist directory: /home/data/chroma_db")
            continue
        
        # Query the RAG system
        print("\n🤔 Thinking...\n")
        
        try:
            print("   [1/3] Embedding query...", end='', flush=True)
            # We'll need to call search directly to add progress
            query_embedding = rag.embed_text(question)
            print(" ✓")
            
            print("   [2/3] Searching vector database...", end='', flush=True)
            results = rag.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            print(" ✓")
            
            # Debug: Show how many results were retrieved
            retrieved_count = len(results["documents"][0]) if results["documents"] else 0
            print(f"   Retrieved {retrieved_count} source(s)")
            
            if retrieved_count == 0:
                print("\n⚠️  WARNING: No relevant documents found! The answer below is NOT based on your documents.\n")
                context = None
            else:
                context = "\n\n---\n\n".join(results["documents"][0])
            
            print("   [3/3] Generating answer...", end='', flush=True)
            answer = rag.generate_response(question, context)
            print(" ✓\n")
            
            # Print answer
            print("💬 Answer:")
            print("─" * 80)
            print(answer)
            print("─" * 80)
            
            # Print sources if enabled
            if show_sources and retrieved_count > 0:
                print_sources(results)
            elif show_sources and retrieved_count == 0:
                print("\n📚 Sources used: (none - database may be empty)")
        
        except requests.exceptions.Timeout:
            print(" ⏱️  TIMEOUT")
            print("❌ The LLM took too long to respond (>120s). Try a simpler question or check if Ollama is overloaded.")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print("\nTrying to reconnect...")
            try:
                rag = CEEHistoryRAG()
                print("✓ Reconnected")
            except:
                print("❌ Could not reconnect. Check if Ollama is running.")

if __name__ == "__main__":
    main()
