import pandas as pd
import io

def extract_data_from_csv(file_path: str) -> str:
    """
    Extracts a summary of the CSV data for LLM context.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Get basic info
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        
        # Get data types
        dtypes = df.dtypes.to_dict()
        dtypes_str = ", ".join([f"{col} ({dtype})" for col, dtype in dtypes.items()])
        
        # Get statistical summary for numeric columns
        summary = df.describe(include='all').to_string()
        
        # Get head
        head = df.head(10).to_string()
        
        text_summary = f"""
CSV Data Summary:
- Filename: {file_path.split('/')[-1]}
- Shape: {num_rows} rows x {num_cols} columns
- Columns: {', '.join(columns)}
- Data Types: {dtypes_str}

Statistical Summary:
{summary}

Sample Data (First 10 rows):
{head}
"""
        return text_summary
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def get_csv_as_json(file_path: str, limit: int = 100):
    """
    Returns CSV data as a list of dicts for more complex analysis if needed.
    """
    try:
        df = pd.read_csv(file_path)
        return df.head(limit).to_dict(orient='records')
    except:
        return []
