"""
Table RAG Indexer - Embedding and Vector Database Integration
Supports multiple embedding models and vector databases
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from table_representation import DocumentChunk


@dataclass
class EmbeddedChunk:
    """A document chunk with its embedding"""
    chunk: DocumentChunk
    embedding: np.ndarray
    metadata: Dict[str, Any]


class EmbeddingModel:
    """Base class for embedding models"""
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        raise NotImplementedError
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        self.dimension = None
        
        try:
            import openai
            self.client = openai.OpenAI()
            
            # Set dimensions based on model
            if "small" in model_name:
                self.dimension = 1536
            elif "large" in model_name:
                self.dimension = 3072
            else:
                self.dimension = 1536
        except ImportError:
            print("openai not installed. Install with: pip install openai")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [np.array(item.embedding) for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    def get_dimension(self) -> int:
        return self.dimension


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformers embedding model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not initialized")
        
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.model.get_embedding_dimension())
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not initialized")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb for emb in embeddings]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            dim = self.model.get_embedding_dimension()
            return [np.zeros(dim) for _ in texts]
    
    def get_dimension(self) -> int:
        if self.model is None:
            return 384  # Default for all-MiniLM-L6-v2
        return self.model.get_embedding_dimension()


class VectorDatabase:
    """Base class for vector databases"""
    
    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """Add embedded chunks to the database"""
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Search for similar chunks"""
        raise NotImplementedError
    
    def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by ID"""
        raise NotImplementedError


class ChromaDB(VectorDatabase):
    """ChromaDB implementation"""
    
    def __init__(self, collection_name: str = "table_rag", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            print("chromadb not installed. Install with: pip install chromadb")
    
    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """Add embedded chunks to ChromaDB"""
        if self.collection is None:
            return False
        
        try:
            ids = [chunk.chunk.chunk_id for chunk in embedded_chunks]
            texts = [chunk.chunk.content for chunk in embedded_chunks]
            embeddings = [chunk.embedding.tolist() for chunk in embedded_chunks]
            metadatas = [chunk.chunk.metadata for chunk in embedded_chunks]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding chunks to ChromaDB: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Search for similar chunks in ChromaDB"""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters
            )
            
            chunks = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    chunk = DocumentChunk(
                        chunk_id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        table_id=results['metadatas'][0][i].get('table_id'),
                        row_ids=results['metadatas'][0][i].get('row_ids')
                    )
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error searching in ChromaDB: {e}")
            return []
    
    def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from ChromaDB"""
        if self.collection is None:
            return False
        
        try:
            self.collection.delete(ids=chunk_ids)
            return True
        except Exception as e:
            print(f"Error deleting chunks from ChromaDB: {e}")
            return False


class FAISSIndex(VectorDatabase):
    """FAISS-based local vector index"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks = {}  # Map chunk_id to DocumentChunk
        self.chunk_order = []  # Maintain order of chunks for FAISS indexing
        
        try:
            import faiss
            self.index = faiss.IndexFlatL2(dimension)
        except ImportError:
            print("faiss not installed. Install with: pip install faiss-cpu")
    
    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """Add embedded chunks to FAISS index"""
        if self.index is None:
            return False
        
        try:
            embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
            self.index.add(embeddings)
            
            # Store chunks in order
            for chunk in embedded_chunks:
                self.chunks[chunk.chunk.chunk_id] = chunk.chunk
                self.chunk_order.append(chunk.chunk.chunk_id)
            
            return True
        except Exception as e:
            print(f"Error adding chunks to FAISS: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Search for similar chunks in FAISS"""
        if self.index is None:
            return []
        
        try:
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k)
            
            chunks = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.chunk_order):
                    # Get chunk by index using the maintained order
                    chunk_id = self.chunk_order[idx]
                    chunk = self.chunks[chunk_id]
                    
                    # Apply filters if provided
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if chunk.metadata.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error searching in FAISS: {e}")
            return []
    
    def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from FAISS (requires rebuilding index)"""
        # FAISS doesn't support deletion well, so we'd need to rebuild
        print("Warning: FAISS deletion requires index rebuild")
        return False


class TableRAGIndexer:
    """Main indexer for table RAG system"""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_db: VectorDatabase):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> bool:
        """Index document chunks"""
        try:
            # Process in batches
            embedded_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk.content for chunk in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.embed_texts(texts)
                
                # Create embedded chunks
                for chunk, embedding in zip(batch, embeddings):
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        metadata=chunk.metadata
                    )
                    embedded_chunks.append(embedded_chunk)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Add to vector database
            success = self.vector_db.add_chunks(embedded_chunks)
            
            if success:
                print(f"Successfully indexed {len(embedded_chunks)} chunks")
            
            return success
        
        except Exception as e:
            print(f"Error indexing chunks: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Search for relevant chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search vector database
            results = self.vector_db.search(query_embedding, top_k, filters)
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs"""
        return self.vector_db.delete(chunk_ids)


class EmbeddingModelFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> EmbeddingModel:
        """Create an embedding model by type"""
        models = {
            'openai': OpenAIEmbedding,
            'sentence_transformers': SentenceTransformerEmbedding
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
        
        return models[model_type](**kwargs)


class VectorDatabaseFactory:
    """Factory for creating vector databases"""
    
    @staticmethod
    def create_database(db_type: str, **kwargs) -> VectorDatabase:
        """Create a vector database by type"""
        databases = {
            'chroma': ChromaDB,
            'faiss': FAISSIndex
        }
        
        if db_type not in databases:
            raise ValueError(f"Unknown database type: {db_type}. Available: {list(databases.keys())}")
        
        return databases[db_type](**kwargs)


def main():
    """Example usage"""
    from table_representation import DocumentChunk
    
    # Create sample chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            content="Table with employee data showing salaries and departments",
            metadata={"table_id": "employees", "row_count": 5}
        ),
        DocumentChunk(
            chunk_id="chunk_2", 
            content="Employee Alice works in Engineering with salary 50000",
            metadata={"table_id": "employees", "row_index": 0}
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            content="Employee Bob works in Sales with salary 60000",
            metadata={"table_id": "employees", "row_index": 1}
        )
    ]
    
    # Try with sentence transformers (local, free)
    try:
        print("Creating Sentence Transformer embedding model...")
        embedding_model = EmbeddingModelFactory.create_model('sentence_transformers')
        
        print("Creating FAISS vector database...")
        vector_db = VectorDatabaseFactory.create_database('faiss', 
                                                        dimension=embedding_model.get_dimension())
        
        print("Creating Table RAG Indexer...")
        indexer = TableRAGIndexer(embedding_model, vector_db)
        
        print("Indexing chunks...")
        indexer.index_chunks(chunks)
        
        print("\nSearching for 'engineering salary'...")
        results = indexer.search("engineering salary", top_k=2)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {result.content}")
            print(f"Metadata: {result.metadata}")
    
    except Exception as e:
        print(f"Error in example: {e}")
        print("Make sure to install required packages: pip install sentence-transformers faiss-cpu")


if __name__ == "__main__":
    main()