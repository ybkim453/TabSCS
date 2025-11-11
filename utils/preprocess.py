# Normalizer & Preprocess
import pandas as pd

## Adapted from Binder and Dater paper 

def dict2df(table):
    """Convert dictionary to DataFrame (utils.preprocess)"""
    header, rows = table[0], table[1:]

    seen = {}
    unique_header = []
    for i, col in enumerate(header):
        col_name = str(col).strip().strip('"').strip("'")
        if not col_name or col_name.lower() == "nan":
            col_name = f"Unnamed: {i}" # Empty header when Unnamed : i

        if col_name not in seen:
            seen[col_name] = 0
            unique_header.append(col_name)
        else:
            seen[col_name] += 1
            unique_header.append(f"{col_name}_{seen[col_name]}") # When duplicate headers, name_i

    df = pd.DataFrame(data=rows, columns=unique_header)
    
    for col in df.columns:
    # Remove commas for all values
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)

        # Handle empty strings or 'nan' like values as missing
        df[col] = df[col].replace({"": None, "nan": None, "NaN": None})

        # Convert values that can be converted to numbers to numbers
        df[col] = pd.to_numeric(df[col], errors="ignore")

        if df[col].dtype == object and df[col].str.contains(r"\d{1,2}\s+\w+", na=False).any():
            df[col] = pd.to_datetime(df[col], errors="ignore", dayfirst=True)

    return df

def table_linearization(table: pd.DataFrame, style: str = 'pipe'):
    """Convert table to pipe-formatted string (utils.preprocess)"""
    linear_table = ''
    if style == 'pipe':
        header = ' | '.join(table.columns) + '\n'
        linear_table += header
        rows = table.values.tolist()
        for row_idx, row in enumerate(rows):
            line = ' | '.join(str(v) for v in row)
            if row_idx != len(rows) - 1:
                line += '\n'
            linear_table += line
    return linear_table

def convert_df_type(df):
    """Convert DataFrame type (utils.normalizer)"""
    # 원본은 복잡하니까 간단 버전만
    return df