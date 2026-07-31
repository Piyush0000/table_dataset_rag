"""
Main Table Extraction Module for RAG
Supports extraction from PDF, HTML, and DOCX documents
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path


@dataclass
class Table:
    """Represents an extracted table with metadata"""
    table_id: str
    content: pd.DataFrame
    page_number: Optional[int] = None
    caption: Optional[str] = None
    context: Optional[str] = None
    headers: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = self.content.columns.tolist()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format"""
        return {
            'table_id': self.table_id,
            'page_number': self.page_number,
            'caption': self.caption,
            'context': self.context,
            'headers': self.headers,
            'data': self.content.to_dict(orient='records'),
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        """Convert table to JSON format"""
        return json.dumps(self.to_dict(), indent=2, default=str)


class TableExtractor:
    """Main table extraction class that delegates to format-specific extractors"""
    
    def __init__(self):
        self.extractors = {
            '.pdf': PDFTableExtractor(),
            '.html': HTMLTableExtractor(),
            '.htm': HTMLTableExtractor(),
            '.docx': DOCXTableExtractor()
        }
    
    def extract_from_file(self, file_path: str) -> List[Table]:
        """Extract tables from a file based on its extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.extractors:
            raise ValueError(f"Unsupported file format: {extension}")
        
        extractor = self.extractors[extension]
        return extractor.extract(file_path)
    
    def extract_from_directory(self, directory_path: str) -> Dict[str, List[Table]]:
        """Extract tables from all supported files in a directory"""
        path = Path(directory_path)
        results = {}
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.extractors:
                try:
                    tables = self.extract_from_file(str(file_path))
                    if tables:
                        results[str(file_path)] = tables
                except Exception as e:
                    print(f"Error extracting from {file_path}: {e}")
        
        return results


class PDFTableExtractor:
    """Extract tables from PDF documents using pdfplumber and camelot"""
    
    def __init__(self):
        self.use_pdfplumber = True
        self.use_camelot = False
        
        # Try to import libraries
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            print("pdfplumber not installed. Install with: pip install pdfplumber")
            self.use_pdfplumber = False
        
        try:
            import camelot
            self.camelot = camelot
        except ImportError:
            print("camelot not installed. Install with: pip install camelot-py[cv]")
            self.use_camelot = False
    
    def extract(self, file_path: str) -> List[Table]:
        """Extract tables from PDF file"""
        tables = []
        
        if self.use_pdfplumber:
            tables.extend(self._extract_with_pdfplumber(file_path))
        
        if self.use_camelot:
            tables.extend(self._extract_with_camelot(file_path))
        
        return tables
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Table]:
        """Extract tables using pdfplumber"""
        tables = []
        
        try:
            with self.pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    extracted_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(extracted_tables):
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        # Clean the DataFrame
                        df = self._clean_dataframe(df)
                        
                        table = Table(
                            table_id=f"pdf_page_{page_num}_table_{table_idx}",
                            content=df,
                            page_number=page_num,
                            metadata={'extraction_method': 'pdfplumber'}
                        )
                        tables.append(table)
        
        except Exception as e:
            print(f"Error extracting with pdfplumber: {e}")
        
        return tables
    
    def _extract_with_camelot(self, file_path: str) -> List[Table]:
        """Extract tables using camelot"""
        tables = []
        
        try:
            # Extract all tables
            extracted_tables = self.camelot.read_pdf(file_path, pages='all')
            
            for table_idx, table in enumerate(extracted_tables):
                df = table.df
                df = self._clean_dataframe(df)
                
                table = Table(
                    table_id=f"camelot_table_{table_idx}",
                    content=df,
                    page_number=table.page,
                    metadata={'extraction_method': 'camelot', 'accuracy': table.accuracy}
                )
                tables.append(table)
        
        except Exception as e:
            print(f"Error extracting with camelot: {e}")
        
        return tables
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df


class HTMLTableExtractor:
    """Extract tables from HTML documents"""
    
    def __init__(self):
        try:
            from bs4 import BeautifulSoup
            self.bs4 = BeautifulSoup
        except ImportError:
            print("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            self.bs4 = None
    
    def extract(self, file_path: str) -> List[Table]:
        """Extract tables from HTML file"""
        tables = []
        
        if self.bs4 is None:
            return tables
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = self.bs4(html_content, 'html.parser')
            table_elements = soup.find_all('table')
            
            for table_idx, table_elem in enumerate(table_elements):
                # Extract table caption
                caption = table_elem.find('caption')
                caption_text = caption.get_text(strip=True) if caption else None
                
                # Convert to DataFrame
                try:
                    # Use StringIO to pass HTML string to read_html
                    from io import StringIO
                    df = pd.read_html(StringIO(str(table_elem)))[0]
                    df = self._clean_dataframe(df)
                    
                    table = Table(
                        table_id=f"html_table_{table_idx}",
                        content=df,
                        caption=caption_text,
                        metadata={'extraction_method': 'pandas'}
                    )
                    tables.append(table)
                except Exception as e:
                    print(f"Error parsing HTML table {table_idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error reading HTML file: {e}")
        
        return tables
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.columns = [str(col).strip() for col in df.columns]
        df = df.reset_index(drop=True)
        return df


class DOCXTableExtractor:
    """Extract tables from DOCX documents"""
    
    def __init__(self):
        try:
            from docx import Document
            self.Document = Document
        except ImportError:
            print("python-docx not installed. Install with: pip install python-docx")
            self.Document = None
    
    def extract(self, file_path: str) -> List[Table]:
        """Extract tables from DOCX file"""
        tables = []
        
        if self.Document is None:
            return tables
        
        try:
            doc = self.Document(file_path)
            
            for table_idx, table in enumerate(doc.tables):
                # Extract table data
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if len(table_data) < 2:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                df = self._clean_dataframe(df)
                
                table = Table(
                    table_id=f"docx_table_{table_idx}",
                    content=df,
                    metadata={'extraction_method': 'python-docx'}
                )
                tables.append(table)
        
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
        
        return tables
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.columns = [str(col).strip() for col in df.columns]
        df = df.reset_index(drop=True)
        return df


def main():
    """Example usage"""
    extractor = TableExtractor()
    
    # Example: Extract from the PDF in the current directory
    pdf_path = "datasets_vipul_Sir (1).pdf"
    if os.path.exists(pdf_path):
        print(f"Extracting tables from {pdf_path}...")
        tables = extractor.extract_from_file(pdf_path)
        print(f"Found {len(tables)} tables")
        
        for table in tables:
            print(f"\nTable ID: {table.table_id}")
            print(f"Shape: {table.content.shape}")
            print(f"Headers: {table.headers}")
            print(f"First few rows:")
            print(table.content.head())
    else:
        print(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()