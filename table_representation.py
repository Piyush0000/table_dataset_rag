"""
Table Representation Strategies for RAG
Implements row-level, table-level, and hybrid chunking strategies
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from table_processor import TableProcessor, TableSchema


@dataclass
class DocumentChunk:
    """Represents a chunk of content for RAG"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    table_id: Optional[str] = None
    row_ids: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata,
            'table_id': self.table_id,
            'row_ids': self.row_ids
        }


class TableRepresentationStrategy:
    """Base class for table representation strategies"""
    
    def __init__(self, processor: TableProcessor):
        self.processor = processor
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema, 
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert table to document chunks"""
        raise NotImplementedError


class RowLevelStrategy(TableRepresentationStrategy):
    """Represent each row as a separate document chunk"""
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema,
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert each table row to a document chunk"""
        chunks = []
        
        # Create header context
        header_context = self._create_header_context(df, schema)
        
        for idx, row in df.iterrows():
            # Create row content
            row_content = self._create_row_content(row, header_context, context)
            
            # Create metadata
            metadata = {
                'table_id': table_id,
                'row_index': idx,
                'strategy': 'row_level',
                'total_rows': len(df),
                'columns': df.columns.tolist()
            }
            
            # Add any additional context
            if context:
                metadata['context'] = context
            
            chunk = DocumentChunk(
                chunk_id=f"{table_id}_row_{idx}",
                content=row_content,
                metadata=metadata,
                table_id=table_id,
                row_ids=[str(idx)]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_header_context(self, df: pd.DataFrame, schema: TableSchema) -> str:
        """Create header context for rows"""
        header_info = f"Table with columns: {', '.join(df.columns.tolist())}. "
        
        if schema.primary_keys:
            header_info += f"Primary keys: {', '.join(schema.primary_keys)}. "
        
        if schema.numeric_columns:
            header_info += f"Numeric columns: {', '.join(schema.numeric_columns)}. "
        
        return header_info
    
    def _create_row_content(self, row: pd.Series, header_context: str, 
                           table_context: Optional[str] = None) -> str:
        """Create content for a single row"""
        row_data = []
        for col, value in row.items():
            if pd.notna(value):
                row_data.append(f"{col}: {value}")
        
        row_text = " | ".join(row_data)
        
        content = f"{header_context} Row data: {row_text}"
        
        if table_context:
            content = f"{table_context} {content}"
        
        return content


class TableLevelStrategy(TableRepresentationStrategy):
    """Represent entire table as a single document chunk"""
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema,
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert entire table to a document chunk"""
        # Create table summary
        summary = self.processor.generate_table_summary(df, schema, table_id)
        
        # Create table representation
        table_content = self._create_table_content(df, schema)
        
        # Combine summary and content
        content = f"{summary}\n\n{table_content}"
        
        if context:
            content = f"{context}\n\n{content}"
        
        # Create metadata
        metadata = {
            'table_id': table_id,
            'strategy': 'table_level',
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'primary_keys': schema.primary_keys,
            'data_types': schema.data_types
        }
        
        chunk = DocumentChunk(
            chunk_id=f"{table_id}_full",
            content=content,
            metadata=metadata,
            table_id=table_id,
            row_ids=[str(i) for i in range(len(df))]
        )
        
        return [chunk]
    
    def _create_table_content(self, df: pd.DataFrame, schema: TableSchema) -> str:
        """Create content representation of the table"""
        # Convert to markdown-like format
        lines = []
        
        # Header
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        
        # Data rows (limit to first 20 rows for brevity)
        for idx, row in df.head(20).iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            lines.append("| " + " | ".join(row_values) + " |")
        
        if len(df) > 20:
            lines.append(f"\n... and {len(df) - 20} more rows")
        
        return "\n".join(lines)


class HybridStrategy(TableRepresentationStrategy):
    """Hybrid approach: table summary + individual rows"""
    
    def __init__(self, processor: TableProcessor, include_summary: bool = True):
        super().__init__(processor)
        self.include_summary = include_summary
        self.row_strategy = RowLevelStrategy(processor)
        self.table_strategy = TableLevelStrategy(processor)
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema,
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert table to hybrid representation"""
        chunks = []
        
        # Add table-level summary chunk
        if self.include_summary:
            summary_chunks = self.table_strategy.represent_table(
                table_id, df, schema, context
            )
            # Modify the summary chunk to indicate it's a summary
            for chunk in summary_chunks:
                chunk.metadata['strategy'] = 'hybrid_summary'
                chunk.chunk_id = f"{table_id}_summary"
            chunks.extend(summary_chunks)
        
        # Add row-level chunks
        row_chunks = self.row_strategy.represent_table(
            table_id, df, schema, context
        )
        # Modify row chunks to indicate hybrid strategy
        for chunk in row_chunks:
            chunk.metadata['strategy'] = 'hybrid_row'
            chunk.metadata['has_summary'] = self.include_summary
        chunks.extend(row_chunks)
        
        return chunks


class SemanticChunkingStrategy(TableRepresentationStrategy):
    """Semantic chunking based on content similarity and table structure"""
    
    def __init__(self, processor: TableProcessor, chunk_size: int = 10):
        super().__init__(processor)
        self.chunk_size = chunk_size
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema,
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert table to semantically meaningful chunks"""
        chunks = []
        
        # Group rows into chunks
        num_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, len(df))
            
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Create chunk content
            chunk_content = self._create_chunk_content(
                chunk_df, schema, chunk_idx, num_chunks, context
            )
            
            # Create metadata
            metadata = {
                'table_id': table_id,
                'strategy': 'semantic_chunking',
                'chunk_index': chunk_idx,
                'total_chunks': num_chunks,
                'row_start': start_idx,
                'row_end': end_idx - 1,
                'row_count': len(chunk_df),
                'columns': df.columns.tolist()
            }
            
            chunk = DocumentChunk(
                chunk_id=f"{table_id}_chunk_{chunk_idx}",
                content=chunk_content,
                metadata=metadata,
                table_id=table_id,
                row_ids=[str(i) for i in range(start_idx, end_idx)]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_content(self, chunk_df: pd.DataFrame, schema: TableSchema,
                             chunk_idx: int, total_chunks: int,
                             context: Optional[str] = None) -> str:
        """Create content for a semantic chunk"""
        header_info = f"Table chunk {chunk_idx + 1}/{total_chunks}. "
        header_info += f"Columns: {', '.join(chunk_df.columns.tolist())}. "
        
        # Create table representation
        lines = [header_info]
        lines.append("| " + " | ".join(chunk_df.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(chunk_df.columns)) + " |")
        
        for idx, row in chunk_df.iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            lines.append("| " + " | ".join(row_values) + " |")
        
        content = "\n".join(lines)
        
        if context:
            content = f"{context}\n\n{content}"
        
        return content


class SQLRepresentationStrategy(TableRepresentationStrategy):
    """Represent table as SQL schema and data"""
    
    def represent_table(self, table_id: str, df: pd.DataFrame, schema: TableSchema,
                       context: Optional[str] = None) -> List[DocumentChunk]:
        """Convert table to SQL representation"""
        chunks = []
        
        # Create SQL schema
        sql_schema = self.processor.generate_sql_schema(table_id, schema)
        
        # Create SQL insert statements
        sql_inserts = self._generate_insert_statements(table_id, df)
        
        # Combine schema and data
        content = f"-- Table Schema\n{sql_schema}\n\n-- Sample Data\n{sql_inserts}"
        
        if context:
            content = f"-- Context: {context}\n\n{content}"
        
        # Create metadata
        metadata = {
            'table_id': table_id,
            'strategy': 'sql_representation',
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'data_types': schema.data_types
        }
        
        chunk = DocumentChunk(
            chunk_id=f"{table_id}_sql",
            content=content,
            metadata=metadata,
            table_id=table_id,
            row_ids=[str(i) for i in range(len(df))]
        )
        
        return [chunk]
    
    def _generate_insert_statements(self, table_name: str, df: pd.DataFrame, 
                                   max_rows: int = 50) -> str:
        """Generate SQL INSERT statements"""
        inserts = []
        
        # Limit to max_rows for brevity
        df_sample = df.head(max_rows)
        
        for idx, row in df_sample.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("NULL")
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped_val = val.replace("'", "''")
                    values.append(f"'{escaped_val}'")
                else:
                    values.append(str(val))
            
            insert = f"INSERT INTO {table_name} VALUES ({', '.join(values)});"
            inserts.append(insert)
        
        if len(df) > max_rows:
            inserts.append(f"-- ... and {len(df) - max_rows} more rows")
        
        return "\n".join(inserts)


class TableRepresentationFactory:
    """Factory for creating table representation strategies"""
    
    @staticmethod
    def create_strategy(strategy_name: str, processor: TableProcessor, 
                       **kwargs) -> TableRepresentationStrategy:
        """Create a representation strategy by name"""
        strategies = {
            'row_level': RowLevelStrategy,
            'table_level': TableLevelStrategy,
            'hybrid': HybridStrategy,
            'semantic_chunking': SemanticChunkingStrategy,
            'sql': SQLRepresentationStrategy
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
        
        return strategies[strategy_name](processor, **kwargs)


def main():
    """Example usage"""
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000.50, 60000.00, 70000.75, 55000.25, 65000.00],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize processor
    processor = TableProcessor()
    cleaned_df, schema = processor.process_table(df, "employees")
    
    # Test different strategies
    strategies = ['row_level', 'table_level', 'hybrid', 'semantic_chunking', 'sql']
    
    for strategy_name in strategies:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")
        print('='*50)
        
        strategy = TableRepresentationFactory.create_strategy(strategy_name, processor)
        chunks = strategy.represent_table("employees", cleaned_df, schema)
        
        print(f"Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} (ID: {chunk.chunk_id}):")
            print(f"Content preview: {chunk.content[:200]}...")
            print(f"Metadata: {json.dumps(chunk.metadata, indent=2)}")


if __name__ == "__main__":
    main()