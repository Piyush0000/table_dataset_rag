# Table RAG System

A comprehensive system for extracting tables from documents and making them suitable for Retrieval-Augmented Generation (RAG). This system handles the entire pipeline from table extraction to semantic search.

## Features

- **Multi-format Table Extraction**: Support for PDF, HTML, and DOCX documents
- **Intelligent Table Processing**: Automatic type detection, cleaning, and normalization
- **Multiple Representation Strategies**: Row-level, table-level, hybrid, and semantic chunking
- **Flexible Embedding Models**: Support for OpenAI and Sentence Transformers
- **Vector Database Integration**: ChromaDB and FAISS support
- **Production-Ready**: Modular design with factory patterns for easy customization

## Installation

```bash
# Install basic dependencies
pip install -r requirements.txt

# For PDF extraction with pdfplumber
pip install pdfplumber

# For advanced PDF extraction with camelot (optional)
pip install camelot-py[cv]

# For OpenAI embeddings (optional)
pip install openai

# For ChromaDB instead of FAISS (optional)
pip install chromadb
```

## Quick Start

```python
from table_extractor import TableExtractor
from table_processor import TableProcessor
from table_representation import TableRepresentationFactory
from table_rag_indexer import TableRAGIndexer, EmbeddingModelFactory, VectorDatabaseFactory

# 1. Extract tables from documents
extractor = TableExtractor()
tables = extractor.extract_from_file("document.pdf")

# 2. Process tables (cleaning and type detection)
processor = TableProcessor()
processed_tables = []
for table in tables:
    cleaned_df, schema = processor.process_table(table.content, table.table_id)
    processed_tables.append((cleaned_df, schema))

# 3. Generate representations for RAG
strategy = TableRepresentationFactory.create_strategy('hybrid', processor)
all_chunks = []
for table in tables:
    schema = TableSchema(**table.metadata['schema'])
    chunks = strategy.represent_table(table.table_id, table.content, schema)
    all_chunks.extend(chunks)

# 4. Set up embedding and indexing
embedding_model = EmbeddingModelFactory.create_model('sentence_transformers')
vector_db = VectorDatabaseFactory.create_database('faiss', dimension=384)
indexer = TableRAGIndexer(embedding_model, vector_db)

# 5. Index the chunks
indexer.index_chunks(all_chunks)

# 6. Search for relevant information
results = indexer.search("engineering department budget", top_k=5)
```

## Architecture

### 1. Table Extraction (`table_extractor.py`)

The extraction module supports multiple document formats:

- **PDF Extraction**: Uses `pdfplumber` and optionally `camelot` for complex tables
- **HTML Extraction**: Uses `pandas` and `BeautifulSoup` for HTML tables
- **DOCX Extraction**: Uses `python-docx` for Word document tables

```python
from table_extractor import TableExtractor

extractor = TableExtractor()
tables = extractor.extract_from_file("document.pdf")
# tables = extractor.extract_from_directory("./documents/")
```

### 2. Table Processing (`table_processor.py`)

The processing module handles:

- **Data Cleaning**: Removes empty rows/columns, normalizes text
- **Type Detection**: Automatically detects numeric, date, boolean, and categorical columns
- **Schema Generation**: Creates comprehensive schema with constraints
- **Summary Generation**: Creates natural language summaries of tables

```python
from table_processor import TableProcessor

processor = TableProcessor()
cleaned_df, schema = processor.process_table(df, "table_name")

# Get table summary
summary = processor.generate_table_summary(cleaned_df, schema, "table_name")

# Generate SQL schema
sql_schema = processor.generate_sql_schema("table_name", schema)
```

### 3. Table Representation (`table_representation.py`)

Multiple chunking strategies for RAG:

- **Row-Level**: Each table row becomes a separate document
- **Table-Level**: Entire table as a single document with summary
- **Hybrid**: Combination of table summary + individual rows
- **Semantic Chunking**: Groups rows into meaningful chunks
- **SQL Representation**: Tables represented as SQL schema and data

```python
from table_representation import TableRepresentationFactory

# Create strategy
strategy = TableRepresentationFactory.create_strategy('hybrid', processor)

# Generate chunks
chunks = strategy.represent_table(table_id, df, schema, context)
```

### 4. Embedding & Indexing (`table_rag_indexer.py`)

Support for multiple embedding models and vector databases:

**Embedding Models:**
- `sentence_transformers`: Local, free models (recommended)
- `openai`: OpenAI's embedding API (requires API key)

**Vector Databases:**
- `faiss`: Local, efficient vector search
- `chroma`: Persistent vector database with metadata filtering

```python
from table_rag_indexer import EmbeddingModelFactory, VectorDatabaseFactory, TableRAGIndexer

# Create embedding model
embedding_model = EmbeddingModelFactory.create_model('sentence_transformers')

# Create vector database
vector_db = VectorDatabaseFactory.create_database('faiss', dimension=384)

# Create indexer
indexer = TableRAGIndexer(embedding_model, vector_db)

# Index and search
indexer.index_chunks(chunks)
results = indexer.search("query text", top_k=5)
```

## Configuration Examples

### Using OpenAI Embeddings with ChromaDB

```python
# Create OpenAI embedding model
embedding_model = EmbeddingModelFactory.create_model(
    'openai', 
    model_name='text-embedding-3-small'
)

# Create ChromaDB
vector_db = VectorDatabaseFactory.create_database(
    'chroma',
    collection_name='table_rag',
    persist_directory='./chroma_db'
)

# Create indexer
indexer = TableRAGIndexer(embedding_model, vector_db)
```

### Custom Chunking Strategy

```python
from table_representation import SemanticChunkingStrategy

# Create custom strategy with chunk size of 15 rows
strategy = SemanticChunkingStrategy(processor, chunk_size=15)
chunks = strategy.represent_table(table_id, df, schema)
```

### Batch Processing

```python
# Process multiple files
extractor = TableExtractor()
results = extractor.extract_from_directory("./documents/")

# Process all tables
all_chunks = []
for file_path, tables in results.items():
    for table in tables:
        cleaned_df, schema = processor.process_table(table.content, table.table_id)
        chunks = strategy.represent_table(table.table_id, cleaned_df, schema)
        all_chunks.extend(chunks)

# Index all at once
indexer.index_chunks(all_chunks, batch_size=50)
```

## Use Cases

Based on your research, this system is suitable for:

- **Table RAG**: Semantic search over tabular data
- **Table Question Answering**: Answer questions about table contents
- **Enterprise Search**: Search across business documents with tables
- **Financial Analysis**: Process financial reports and statements
- **Data Analysis**: Natural language interface to structured data

## Advanced Features

### Metadata Filtering

```python
# Search with metadata filters
results = indexer.search(
    "engineering data",
    top_k=5,
    filters={"table_id": "employees", "strategy": "hybrid_row"}
)
```

### SQL Representation

```python
# Generate SQL from tables
strategy = TableRepresentationFactory.create_strategy('sql', processor)
sql_chunks = strategy.represent_table(table_id, df, schema)

# Use for Text-to-SQL applications
```

### Custom Processing Pipeline

```python
# Custom table processing
class CustomProcessor(TableProcessor):
    def clean_table(self, df):
        # Custom cleaning logic
        df = super().clean_table(df)
        # Add custom processing
        return df

processor = CustomProcessor()
```

## Examples

Run the complete example:

```bash
python example_usage.py
```

This will demonstrate:
- Complete pipeline from extraction to retrieval
- Quick start with sample data
- PDF extraction from your research document

## Module Structure

```
table_rag/
├── table_extractor.py      # Multi-format table extraction
├── table_processor.py      # Table processing and normalization
├── table_representation.py # Multiple chunking strategies
├── table_rag_indexer.py    # Embedding and vector database integration
├── example_usage.py        # Complete usage examples
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Performance Considerations

- **Batch Processing**: Process documents in batches for better performance
- **Embedding Caching**: Cache embeddings to avoid recomputation
- **Strategy Selection**: Choose strategy based on use case:
  - Row-level: For detailed row-specific queries
  - Table-level: For table-level questions and summaries
  - Hybrid: Best of both worlds (recommended)
- **Vector Database**: Use FAISS for local use, ChromaDB for production

## Troubleshooting

### PDF Extraction Issues

```bash
# Install system dependencies for camelot
# On Ubuntu/Debian:
sudo apt-get install python3-tk ghostscript

# On macOS:
brew install tcl-tk ghostscript
```

### Memory Issues

```python
# Process in smaller batches
indexer.index_chunks(chunks, batch_size=10)

# Use row-level strategy to reduce chunk size
strategy = TableRepresentationFactory.create_strategy('row_level', processor)
```

### Embedding Model Issues

```python
# Try different sentence transformer models
embedding_model = EmbeddingModelFactory.create_model(
    'sentence_transformers',
    model_name='all-MiniLM-L6-v2'  # Faster, smaller
)
```

## Future Enhancements

Based on your research notes, potential improvements:

- **Table Chunking Strategies**: More sophisticated semantic chunking
- **Table Embeddings**: Specialized embeddings for tabular data
- **Hybrid SQL + Vector Search**: Combine structured and semantic search
- **Schema Retrieval**: Retrieve table schemas along with data
- **Enterprise Architectures**: Multi-tenant, scalable implementations

## Contributing

This is a research implementation based on public Table RAG datasets. Feel free to extend and modify for your specific use cases.

## License

This is a research implementation. Ensure compliance with dataset licenses and embedding model terms of service.

## Acknowledgments

Based on research from public Table RAG datasets including:
- WikiTableQuestions
- WikiSQL  
- Spider
- BIRD
- HybridQA
- FinQA

## References

Your research document: `datasets_vipul_Sir (1).md`

Key insights from your research:
- Most benchmarks use Wikipedia tables
- Spider and BIRD are most widely used for Text-to-SQL
- Enterprise systems need SQL + metadata + semantic retrieval
- Financial datasets focus on numerical reasoning
- Hybrid approaches are closer to real-world applications"# table_dataset_rag" 
