import pandas as pd
from src.config.db_config import get_db_connection

def perform_data_analysis():
    """Fungsi untuk melakukan analisis data."""
    conn = get_db_connection()

    sql_query = 'SELECT * FROM users'
    df = pd.read_sql_query(sql_query, conn)

    conn.close()

    average_value = df['your_column'].mean()

    analysis_result = {
        'total_rows': len(df),
        'average_value': average_value,
        'data': df.to_dict(orient='records')  
    }

    return analysis_result
