"""
Database utilities for storing predictions and user data
"""
import sqlite3
import pandas as pd
from datetime import datetime
import json
import hashlib

class DatabaseManager:
    def __init__(self, db_path='churn_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                prediction INTEGER,
                probability REAL,
                confidence REAL,
                model_used TEXT,
                input_features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT
            )
        ''')
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT DEFAULT 'viewer',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                auc_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # A/B test results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                model_a TEXT,
                model_b TEXT,
                customer_id TEXT,
                model_used TEXT,
                prediction INTEGER,
                probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    
    def save_prediction(self, customer_id, prediction, probability, confidence, model_used, input_features, user_id='anonymous'):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (customer_id, prediction, probability, confidence, model_used, input_features, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (customer_id, prediction, probability, confidence, model_used, json.dumps(input_features), user_id))
        
        conn.commit()
        conn.close()
    
    def get_predictions_history(self, limit=100):
        """Get recent predictions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df
    
    def get_prediction_stats(self):
        """Get prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Total predictions
        total = pd.read_sql_query('SELECT COUNT(*) as count FROM predictions', conn).iloc[0]['count']
        
        # Churn rate
        churn_rate = pd.read_sql_query('''
            SELECT AVG(CAST(prediction AS FLOAT)) as churn_rate FROM predictions
        ''', conn).iloc[0]['churn_rate']
        
        # Model usage
        model_usage = pd.read_sql_query('''
            SELECT model_used, COUNT(*) as count 
            FROM predictions 
            GROUP BY model_used
        ''', conn)
        
        # Daily predictions
        daily_preds = pd.read_sql_query('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        ''', conn)
        
        conn.close()
        
        return {
            'total_predictions': total,
            'churn_rate': churn_rate or 0,
            'model_usage': model_usage.to_dict('records'),
            'daily_predictions': daily_preds.to_dict('records')
        }
    
    def create_user(self, username, password, role='viewer'):
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
            ''', (username, password_hash, role))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('''
            SELECT * FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        
        if user:
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP 
                WHERE username = ?
            ''', (username,))
            conn.commit()
        
        conn.close()
        return user is not None
    
    def save_ab_test_result(self, test_name, model_a, model_b, customer_id, model_used, prediction, probability):
        """Save A/B test result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_tests 
            (test_name, model_a, model_b, customer_id, model_used, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (test_name, model_a, model_b, customer_id, model_used, prediction, probability))
        
        conn.commit()
        conn.close()

# Global database manager
db_manager = DatabaseManager()