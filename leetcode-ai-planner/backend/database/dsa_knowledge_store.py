"""
DSA Knowledge RAG system using local PDFs
"""
import os
import re
import pickle
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from config.settings import DATA_DIR, EMBEDDING_MODEL_NAME


class DSAKnowledgeStore:
    """RAG system for DSA knowledge from local PDFs"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index_path = os.path.join(DATA_DIR, "dsa_knowledge_index.bin")
        self.metadata_path = os.path.join(DATA_DIR, "dsa_knowledge_metadata.pkl")
        
        # PDF files in data folder
        self.pdf_files = [
            os.path.join(DATA_DIR, "Dsa.pdf"),
            os.path.join(DATA_DIR, "Dsa2.pdf"),
            os.path.join(DATA_DIR, "Dsa3.pdf")
        ]
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("📚 Loading DSA knowledge base...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ DSA Knowledge: {len(self.metadata['chunks'])} chunks loaded")
        else:
            print("📦 Building DSA knowledge base from local PDFs...")
            self._build_knowledge_base()
    
    def _extract_text_from_pdf(self, pdf_path: str, max_pages: int = 150) -> str:
        """Extract text from local PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            total_pages = min(len(reader.pages), max_pages)
            
            print(f"   📄 Extracting {total_pages} pages from {os.path.basename(pdf_path)}...")
            
            for i, page in enumerate(reader.pages[:total_pages]):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"   ⚠️  Error on page {i}: {e}")
                    continue
            
            return text
        except Exception as e:
            print(f"❌ Failed to read {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'Page \d+', '', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks"""
        text = self._clean_text(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                last_period = max(
                    chunk.rfind('. '),
                    chunk.rfind('.\n'),
                    chunk.rfind('? '),
                    chunk.rfind('! ')
                )
                
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunk = chunk.strip()
            if len(chunk) > 100:  # Only add meaningful chunks
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _build_knowledge_base(self):
        """Build vector database from local PDFs"""
        all_chunks = []
        all_sources = []
        all_page_refs = []
        
        for pdf_path in self.pdf_files:
            if not os.path.exists(pdf_path):
                print(f"⚠️  PDF not found: {pdf_path}")
                continue
            
            print(f"📖 Processing: {os.path.basename(pdf_path)}...")
            
            # Extract text
            text = self._extract_text_from_pdf(pdf_path)
            
            if not text:
                print(f"⚠️  No text extracted from {os.path.basename(pdf_path)}")
                continue
            
            # Create chunks
            chunks = self._chunk_text(text)
            
            # Metadata
            source_name = os.path.basename(pdf_path)
            all_chunks.extend(chunks)
            all_sources.extend([source_name] * len(chunks))
            all_page_refs.extend([f"Chunk {i+1}" for i in range(len(chunks))])
            
            print(f"✅ Processed {source_name}: {len(chunks)} chunks")
        
        if not all_chunks:
            raise Exception("No content extracted from PDFs. Check if PDFs are readable.")
        
        print(f"\n🔢 Creating embeddings for {len(all_chunks)} chunks...")
        
        # Create embeddings with progress
        embeddings = self.embedding_model.encode(
            all_chunks, 
            show_progress_bar=True, 
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save index and metadata
        print("💾 Saving index and metadata...")
        faiss.write_index(self.index, self.index_path)
        
        self.metadata = {
            "chunks": all_chunks,
            "sources": all_sources,
            "page_refs": all_page_refs
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✅ DSA Knowledge Base built successfully!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Index size: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant DSA knowledge"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < len(self.metadata["chunks"]):
                results.append({
                    "content": self.metadata["chunks"][idx],
                    "source": self.metadata["sources"][idx],
                    "page_ref": self.metadata["page_refs"][idx],
                    "similarity": float(similarities[0][i])
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        if not hasattr(self, 'metadata'):
            return {"status": "not loaded"}
        
        return {
            "total_chunks": len(self.metadata["chunks"]),
            "sources": list(set(self.metadata["sources"])),
            "index_size": self.index.ntotal if hasattr(self, 'index') else 0
        }


# Singleton instance
try:
    dsa_knowledge_store = DSAKnowledgeStore()
except Exception as e:
    print(f"❌ Failed to initialize DSA knowledge store: {e}")
    dsa_knowledge_store = None
