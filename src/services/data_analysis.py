import pandas as pd
from psycopg2.extras import RealDictCursor
from src.config.db_config import get_db_connection

def perform_data_analysis():
    """Fungsi untuk melakukan analisis data."""
    engine = get_db_connection()

    sql_query = 'SELECT * FROM users'
    df = pd.read_sql_query(sql_query, engine)

    analysis_result = {
        'total_rows': len(df),
        'data': df.to_dict(orient='records')  
    }

    return analysis_result

def fetch_data():
    """Fungsi untuk mengambil data dari tabel users tanpa menggunakan pandas."""
    conn = None
    data = []

    try:
        conn = get_db_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute('SELECT * FROM users')
            data = cursor.fetchall()  

    except Exception as e:
        print(f"Error fetching data: {e}")

    finally:
        if conn:
            conn.close()  

    return data