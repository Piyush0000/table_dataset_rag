"""
Complete Example Usage of Table RAG System
Demonstrates the entire pipeline from extraction to retrieval
"""

import os
import pandas as pd
from table_extractor import TableExtractor, Table
from table_processor import TableProcessor, TableSchema
from table_representation import TableRepresentationFactory
from table_rag_indexer import TableRAGIndexer, EmbeddingModelFactory, VectorDatabaseFactory


def create_sample_documents():
    """Create sample documents for testing"""
    # Create sample HTML with tables
    html_content = """
    <html>
    <body>
        <h1>Employee Data</h1>
        <table>
            <caption>Employee Information 2024</caption>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Department</th>
                <th>Salary</th>
            </tr>
            <tr>
                <td>1</td>
                <td>Alice Johnson</td>
                <td>Engineering</td>
                <td>75000</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Bob Smith</td>
                <td>Sales</td>
                <td>65000</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Charlie Brown</td>
                <td>Engineering</td>
                <td>80000</td>
            </tr>
        </table>
        
        <h2>Department Budget</h2>
        <table>
            <tr>
                <th>Department</th>
                <th>Budget</th>
                <th>Headcount</th>
            </tr>
            <tr>
                <td>Engineering</td>
                <td>500000</td>
                <td>15</td>
            </tr>
            <tr>
                <td>Sales</td>
                <td>350000</td>
                <td>10</td>
            </tr>
            <tr>
                <td>Marketing</td>
                <td>200000</td>
                <td>8</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    with open("sample_employees.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("Created sample HTML document: sample_employees.html")
    return "sample_employees.html"


def complete_pipeline_example():
    """Complete example of the Table RAG pipeline"""
    
    print("="*60)
    print("TABLE RAG SYSTEM - COMPLETE PIPELINE EXAMPLE")
    print("="*60)
    
    # Step 1: Create sample document
    print("\n[Step 1] Creating sample document...")
    html_file = create_sample_documents()
    
    # Step 2: Extract tables
    print("\n[Step 2] Extracting tables from document...")
    extractor = TableExtractor()
    tables = extractor.extract_from_file(html_file)
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}: {table.table_id}")
        print(f"Shape: {table.content.shape}")
        print(f"Columns: {table.headers}")
        print(f"First few rows:")
        print(table.content.head())
    
    # Step 3: Process tables
    print("\n[Step 3] Processing tables (cleaning and type detection)...")
    processor = TableProcessor()
    processed_tables = []
    
    for table in tables:
        cleaned_df, schema = processor.process_table(table.content, table.table_id)
        
        # Update table with processed data
        table.content = cleaned_df
        table.metadata['schema'] = schema.__dict__
        
        processed_tables.append(table)
        
        print(f"\nProcessed table: {table.table_id}")
        print(f"Data types: {schema.data_types}")
        print(f"Primary keys: {schema.primary_keys}")
        print(f"Summary: {processor.generate_table_summary(cleaned_df, schema, table.table_id)}")
    
    # Step 4: Generate representations
    print("\n[Step 4] Generating table representations for RAG...")
    
    # Try different strategies
    strategies = ['hybrid', 'row_level', 'table_level']
    all_chunks = []
    
    for strategy_name in strategies:
        print(f"\nUsing strategy: {strategy_name}")
        strategy = TableRepresentationFactory.create_strategy(strategy_name, processor)
        
        for table in processed_tables:
            schema = TableSchema(**table.metadata['schema'])
            chunks = strategy.represent_table(
                table.table_id, 
                table.content, 
                schema,
                table.caption
            )
            all_chunks.extend(chunks)
            print(f"  Generated {len(chunks)} chunks from {table.table_id}")
    
    print(f"\nTotal chunks generated: {len(all_chunks)}")
    
    # Step 5: Set up embedding and indexing
    print("\n[Step 5] Setting up embedding model and vector database...")
    
    try:
        # Use sentence transformers (local, free option)
        embedding_model = EmbeddingModelFactory.create_model('sentence_transformers')
        print(f"Created embedding model with dimension: {embedding_model.get_dimension()}")
        
        # Use FAISS for local vector storage
        vector_db = VectorDatabaseFactory.create_database(
            'faiss', 
            dimension=embedding_model.get_dimension()
        )
        print("Created FAISS vector database")
        
        # Create indexer
        indexer = TableRAGIndexer(embedding_model, vector_db)
        
        # Step 6: Index chunks
        print("\n[Step 6] Indexing chunks...")
        success = indexer.index_chunks(all_chunks)
        
        if success:
            print("Successfully indexed all chunks")
            
            # Step 7: Search examples
            print("\n[Step 7] Testing search functionality...")
            
            test_queries = [
                "engineering department budget",
                "employee with highest salary",
                "sales department headcount",
                "Alice Johnson information"
            ]
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                results = indexer.search(query, top_k=2)
                
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"\n  Result {i+1}:")
                    print(f"  Content: {result.content[:200]}...")
                    print(f"  Table ID: {result.metadata.get('table_id')}")
                    print(f"  Strategy: {result.metadata.get('strategy')}")
        else:
            print("Failed to index chunks")
    
    except Exception as e:
        print(f"Error in indexing step: {e}")
        print("Make sure to install required packages:")
        print("pip install sentence-transformers faiss-cpu")
    
    # Cleanup
    if os.path.exists(html_file):
        os.remove(html_file)
        print(f"\nCleaned up sample file: {html_file}")


def quick_start_example():
    """Quick start example with minimal setup"""
    
    print("\n" + "="*60)
    print("QUICK START EXAMPLE")
    print("="*60)
    
    # Create sample data directly
    sample_data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'price': [999.99, 29.99, 79.99, 299.99, 149.99],
        'stock': [50, 200, 150, 75, 100]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nSample Product Data:")
    print(df)
    
    # Process the table
    processor = TableProcessor()
    cleaned_df, schema = processor.process_table(df, "products")
    
    print("\nProcessed Schema:")
    print(f"Columns: {schema.columns}")
    print(f"Data Types: {schema.data_types}")
    print(f"Numeric Columns: {schema.numeric_columns}")
    
    # Generate representation
    strategy = TableRepresentationFactory.create_strategy('row_level', processor)
    chunks = strategy.represent_table("products", cleaned_df, schema)
    
    print(f"\nGenerated {len(chunks)} chunks")
    print("\nFirst chunk content:")
    print(chunks[0].content)
    
    # Simple text-based search (without embeddings)
    print("\nSimple text search for 'laptop':")
    for chunk in chunks:
        if 'laptop' in chunk.content.lower():
            print(f"Found in chunk {chunk.chunk_id}")
            print(f"Content: {chunk.content}")


def pdf_extraction_example():
    """Example specifically for PDF extraction"""
    
    print("\n" + "="*60)
    print("PDF EXTRACTION EXAMPLE")
    print("="*60)
    
    pdf_file = "datasets_vipul_Sir (1).pdf"
    
    if os.path.exists(pdf_file):
        print(f"\nExtracting tables from: {pdf_file}")
        
        extractor = TableExtractor()
        try:
            tables = extractor.extract_from_file(pdf_file)
            print(f"Found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                print(f"\nTable {i+1}:")
                print(f"  ID: {table.table_id}")
                print(f"  Page: {table.page_number}")
                print(f"  Shape: {table.content.shape}")
                print(f"  Columns: {table.headers[:5]}...")  # Show first 5 columns
                
                if table.content.shape[0] > 0:
                    print(f"  Sample data:")
                    print(table.content.head(2))
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            print("Note: PDF extraction requires pdfplumber. Install with: pip install pdfplumber")
    else:
        print(f"PDF file not found: {pdf_file}")


def main():
    """Run all examples"""
    
    print("TABLE RAG SYSTEM - EXAMPLES")
    print("="*60)
    
    # Run complete pipeline example
    complete_pipeline_example()
    
    # Run quick start example
    quick_start_example()
    
    # Run PDF extraction example
    pdf_extraction_example()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()