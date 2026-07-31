"""
Table Processing and Normalization Module
Handles cleaning, type detection, semantic analysis, and summary generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import warnings
from dataclasses import dataclass
import json


@dataclass
class TableSchema:
    """Represents the schema and semantic information of a table"""
    columns: List[str]
    data_types: Dict[str, str]
    primary_keys: List[str]
    foreign_keys: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    constraints: Dict[str, Any]


class TableProcessor:
    """Process and normalize extracted tables"""
    
    def __init__(self):
        self.type_mapping = {
            'int64': 'integer',
            'float64': 'float',
            'datetime64': 'date',
            'object': 'string',
            'bool': 'boolean'
        }
    
    def process_table(self, df: pd.DataFrame, table_name: str = "table") -> Tuple[pd.DataFrame, TableSchema]:
        """Process a table and return cleaned DataFrame with schema"""
        # Clean the table
        cleaned_df = self.clean_table(df)
        
        # Detect data types
        data_types = self.detect_data_types(cleaned_df)
        
        # Convert to appropriate types
        cleaned_df = self.convert_types(cleaned_df, data_types)
        
        # Generate schema
        schema = self.generate_schema(cleaned_df, table_name, data_types)
        
        return cleaned_df, schema
    
    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize table data"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = self._clean_column_names(df.columns)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(self._clean_text_value)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean column names"""
        cleaned = []
        for col in columns:
            # Convert to string and strip
            col = str(col).strip()
            # Remove special characters except underscores
            col = re.sub(r'[^\w\s]', '', col)
            # Replace spaces with underscores
            col = re.sub(r'\s+', '_', col)
            # Remove leading/trailing underscores
            col = col.strip('_')
            # Ensure non-empty
            if not col:
                col = f"column_{len(cleaned)}"
            cleaned.append(col)
        return cleaned
    
    def _clean_text_value(self, value: Any) -> str:
        """Clean individual text values"""
        if pd.isna(value):
            return ""
        value = str(value).strip()
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value)
        return value
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect semantic data types for each column"""
        data_types = {}
        
        for col in df.columns:
            # Skip if column is empty
            if df[col].empty or df[col].isna().all():
                data_types[col] = 'string'
                continue
            
            # Try to detect numeric
            if self._is_numeric_column(df[col]):
                if self._is_integer_column(df[col]):
                    data_types[col] = 'integer'
                else:
                    data_types[col] = 'float'
            # Try to detect date
            elif self._is_date_column(df[col]):
                data_types[col] = 'date'
            # Try to detect boolean
            elif self._is_boolean_column(df[col]):
                data_types[col] = 'boolean'
            else:
                data_types[col] = 'string'
        
        return data_types
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if column contains numeric data"""
        try:
            # Convert to numeric, coercing errors
            numeric_series = pd.to_numeric(series, errors='coerce')
            # If more than 70% are numeric, consider it numeric
            return numeric_series.notna().sum() / len(series) > 0.7
        except:
            return False
    
    def _is_integer_column(self, series: pd.Series) -> bool:
        """Check if numeric column contains integers"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Check if non-null values are integers
            non_null = numeric_series.dropna()
            if len(non_null) == 0:
                return False
            return (non_null == non_null.astype(int)).all()
        except:
            return False
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if column contains date data"""
        try:
            # First check if column might contain dates (heuristic)
            # Look for common date patterns in the string values
            sample_values = series.dropna().head(10).astype(str)
            
            # Common date patterns
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{1,2}\s+\w+\s+\d{4}',  # D Month YYYY
            ]
            
            import re
            date_like_count = 0
            for val in sample_values:
                for pattern in date_patterns:
                    if re.search(pattern, val):
                        date_like_count += 1
                        break
            
            # If fewer than 30% look like dates, don't even try parsing
            if len(sample_values) > 0 and date_like_count / len(sample_values) < 0.3:
                return False
            
            # Try parsing as datetime with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                date_series = pd.to_datetime(series, errors='coerce')
            
            # If more than 50% are valid dates, consider it date
            return date_series.notna().sum() / len(series) > 0.5
        except:
            return False
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if column contains boolean data"""
        # Get unique non-null values
        unique_values = series.dropna().unique()
        if len(unique_values) > 4:
            return False
        
        # Check for common boolean representations
        bool_patterns = [
            {'true', 'false', 'yes', 'no'},
            {'t', 'f', 'y', 'n'},
            {'1', '0'},
            {'true', 'false'},
            {'yes', 'no'}
        ]
        
        value_set = {str(v).lower() for v in unique_values}
        return any(value_set.issubset(pattern) for pattern in bool_patterns)
    
    def convert_types(self, df: pd.DataFrame, data_types: Dict[str, str]) -> pd.DataFrame:
        """Convert columns to their detected types"""
        df = df.copy()
        
        for col, dtype in data_types.items():
            if col not in df.columns:
                continue
            
            try:
                if dtype == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'date':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'boolean':
                    df[col] = self._convert_to_boolean(df[col])
            except Exception as e:
                print(f"Error converting column {col} to {dtype}: {e}")
        
        return df
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert column to boolean"""
        # Define boolean mappings
        true_values = ['true', 'yes', 'y', 't', '1']
        false_values = ['false', 'no', 'n', 'f', '0']
        
        def convert_value(val):
            if pd.isna(val):
                return pd.NA
            val_lower = str(val).lower()
            if val_lower in true_values:
                return True
            elif val_lower in false_values:
                return False
            else:
                return pd.NA
        
        return series.apply(convert_value)
    
    def generate_schema(self, df: pd.DataFrame, table_name: str, data_types: Dict[str, str]) -> TableSchema:
        """Generate schema information for the table"""
        columns = df.columns.tolist()
        
        # Categorize columns
        numeric_columns = [col for col, dtype in data_types.items() if dtype in ['integer', 'float']]
        categorical_columns = [col for col, dtype in data_types.items() if dtype == 'string']
        date_columns = [col for col, dtype in data_types.items() if dtype == 'date']
        
        # Detect potential primary keys (unique columns)
        primary_keys = self._detect_primary_keys(df)
        
        # Detect foreign keys (heuristic based on naming)
        foreign_keys = self._detect_foreign_keys(df, table_name)
        
        # Generate constraints
        constraints = self._generate_constraints(df, data_types)
        
        return TableSchema(
            columns=columns,
            data_types=data_types,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            date_columns=date_columns,
            constraints=constraints
        )
    
    def _detect_primary_keys(self, df: pd.DataFrame) -> List[str]:
        """Detect potential primary key columns"""
        primary_keys = []
        
        for col in df.columns:
            # Check if column has unique values
            if df[col].nunique() == len(df):
                primary_keys.append(col)
        
        return primary_keys
    
    def _detect_foreign_keys(self, df: pd.DataFrame, table_name: str) -> Dict[str, str]:
        """Detect potential foreign key columns based on naming patterns"""
        foreign_keys = {}
        
        # Common foreign key patterns
        fk_patterns = [
            r'(.*)_id$',
            r'(.*)_code$',
            r'(.*)_key$',
            r'id_(.*)$'
        ]
        
        for col in df.columns:
            for pattern in fk_patterns:
                match = re.match(pattern, col.lower())
                if match:
                    referenced_table = match.group(1)
                    foreign_keys[col] = f"{referenced_table}"
                    break
        
        return foreign_keys
    
    def _generate_constraints(self, df: pd.DataFrame, data_types: Dict[str, str]) -> Dict[str, Any]:
        """Generate column constraints"""
        constraints = {}
        
        for col in df.columns:
            col_constraints = {}
            
            # Range constraints for numeric columns
            if data_types[col] in ['integer', 'float']:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_constraints['min'] = float(non_null.min())
                    col_constraints['max'] = float(non_null.max())
                    col_constraints['mean'] = float(non_null.mean())
            
            # Length constraints for string columns
            elif data_types[col] == 'string':
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_constraints['max_length'] = int(non_null.str.len().max())
                    col_constraints['min_length'] = int(non_null.str.len().min())
            
            # Unique count
            col_constraints['unique_count'] = int(df[col].nunique())
            col_constraints['null_count'] = int(df[col].isna().sum())
            
            constraints[col] = col_constraints
        
        return constraints
    
    def generate_table_summary(self, df: pd.DataFrame, schema: TableSchema, table_name: str = "table") -> str:
        """Generate a natural language summary of the table"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Table '{table_name}' contains {len(df)} rows and {len(df.columns)} columns.")
        
        # Column types
        if schema.numeric_columns:
            summary_parts.append(f"Numeric columns: {', '.join(schema.numeric_columns)}.")
        if schema.categorical_columns:
            summary_parts.append(f"Categorical columns: {', '.join(schema.categorical_columns)}.")
        if schema.date_columns:
            summary_parts.append(f"Date columns: {', '.join(schema.date_columns)}.")
        
        # Primary keys
        if schema.primary_keys:
            summary_parts.append(f"Primary key(s): {', '.join(schema.primary_keys)}.")
        
        # Sample data insights
        if schema.numeric_columns:
            numeric_col = schema.numeric_columns[0]
            non_null = df[numeric_col].dropna()
            if len(non_null) > 0:
                summary_parts.append(f"The {numeric_col} column ranges from {non_null.min():.2f} to {non_null.max():.2f} with an average of {non_null.mean():.2f}.")
        
        return " ".join(summary_parts)
    
    def generate_sql_schema(self, table_name: str, schema: TableSchema) -> str:
        """Generate SQL CREATE TABLE statement"""
        sql_type_mapping = {
            'integer': 'INTEGER',
            'float': 'FLOAT',
            'string': 'TEXT',
            'boolean': 'BOOLEAN',
            'date': 'DATE'
        }
        
        columns_def = []
        for col in schema.columns:
            dtype = schema.data_types[col]
            sql_type = sql_type_mapping.get(dtype, 'TEXT')
            
            col_def = f"    {col} {sql_type}"
            
            # Add PRIMARY KEY constraint
            if col in schema.primary_keys:
                col_def += " PRIMARY KEY"
            
            columns_def.append(col_def)
        
        sql = f"CREATE TABLE {table_name} (\n"
        sql += ",\n".join(columns_def)
        sql += "\n);"
        
        return sql


def main():
    """Example usage"""
    from table_extractor import TableExtractor, Table
    
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000.50, 60000.00, 70000.75, 55000.25, 65000.00],
        'join_date': ['2020-01-15', '2019-05-20', '2021-03-10', '2020-11-25', '2018-07-30'],
        'is_active': ['true', 'true', 'false', 'true', 'true']
    }
    
    df = pd.DataFrame(sample_data)
    
    processor = TableProcessor()
    cleaned_df, schema = processor.process_table(df, "employees")
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nSchema:")
    print(json.dumps(schema.__dict__, indent=2, default=str))
    
    print("\nTable Summary:")
    print(processor.generate_table_summary(cleaned_df, schema, "employees"))
    
    print("\nSQL Schema:")
    print(processor.generate_sql_schema("employees", schema))


if __name__ == "__main__":
    main()