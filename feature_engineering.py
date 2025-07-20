"""
Nexus Bank Financial Analytics Project - Feature Engineering
Author: Nexus Bank Analytics Team
Date: July 2025
Description: Computes technical indicators for market data, aggregates transaction data,
and optimizes database schema for analysis. Ensures all feature values are stored in the database.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
import json
from datetime import datetime
import ta  # Technical Analysis library

# Create directories if they don't exist
LOG_DIR = "logs"
OUTPUT_DIR = "output"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'feature_engineering.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to compute features, optimize database schema, and store all feature values"""
    
    def __init__(self, db_path: str = os.path.join(OUTPUT_DIR, "nexus_bank.db")):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineer")
        self.quality_report = {"features": [], "issues": [], "actions": [], "metrics": {}}

    def compute_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute technical indicators for market data"""
        self.logger.info(f"Computing technical indicators for {symbol}")
        try:
            # Ensure DataFrame is sorted by date
            df = df.sort_index()
            
            # Simple Moving Average (20-day and 50-day)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Average (20-day)
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Bollinger Bands (20-day)
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # Relative Strength Index (14-day)
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Verify no missing feature values
            feature_cols = ['sma_20', 'sma_50', 'ema_20', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'rsi_14']
            missing_counts = df[feature_cols].isna().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    self.logger.warning(f"Found {count} missing values in {symbol}.{col}")
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    self.quality_report['actions'].append(f"Filled {count} missing values in {symbol}.{col}")
            
            # Log metrics
            self.quality_report['features'].append({
                "symbol": symbol,
                "indicators": feature_cols
            })
            self.quality_report['metrics'][symbol] = {
                "rows_processed": len(df),
                "indicators_computed": len(feature_cols),
                "missing_values_filled": missing_counts.to_dict()
            }
            
            return df
        except Exception as e:
            self.logger.error(f"Error computing indicators for {symbol}: {str(e)}")
            self.quality_report['issues'].append(f"Indicator computation error for {symbol}: {str(e)}")
            return df

    def aggregate_transactions(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate transaction data by account and by day"""
        self.logger.info(f"Aggregating transaction data")
        try:
            # Ensure Time is datetime
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Aggregate by account
            account_agg = df.groupby('Account').agg({
                'Amount': ['sum', 'mean', 'count', 'std'],
                'Class': ['sum', 'mean']  # Sum for total frauds, mean for fraud rate
            }).reset_index()
            account_agg.columns = ['Account', 'total_amount', 'avg_amount', 'transaction_count', 
                                 'amount_std', 'total_frauds', 'fraud_rate']
            
            # Handle potential missing values in aggregations
            account_agg = account_agg.fillna(0)
            
            # Aggregate by day
            df['date'] = df['Time'].dt.date
            daily_agg = df.groupby('date').agg({
                'Amount': ['sum', 'mean', 'count', 'std'],
                'Class': ['sum', 'mean']
            }).reset_index()
            daily_agg.columns = ['date', 'total_amount', 'avg_amount', 'transaction_count', 
                               'amount_std', 'total_frauds', 'fraud_rate']
            
            # Handle potential missing values in aggregations
            daily_agg = daily_agg.fillna(0)
            
            # Log metrics
            self.quality_report['metrics']['transactions'] = {
                "account_aggregations": len(account_agg),
                "daily_aggregations": len(daily_agg),
                "original_rows": len(df)
            }
            self.quality_report['features'].append({
                "type": "transaction_aggregates",
                "by_account": ["total_amount", "avg_amount", "transaction_count", "amount_std", 
                              "total_frauds", "fraud_rate"],
                "by_day": ["total_amount", "avg_amount", "transaction_count", "amount_std", 
                          "total_frauds", "fraud_rate"]
            })
            
            return account_agg, daily_agg
        except Exception as e:
            self.logger.error(f"Error aggregating transactions: {str(e)}")
            self.quality_report['issues'].append(f"Transaction aggregation error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def optimize_database(self):
        """Optimize database schema with indexes and partitions"""
        self.logger.info("Optimizing database schema")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create indexes for market data tables
            market_tables = ['market_^GSPC', 'market_EURUSD_X']
            for table in market_tables:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table} (date)")
                self.quality_report['actions'].append(f"Created index idx_{table}_date")
                
            # Create indexes for feature tables
            feature_tables = ['features_^GSPC', 'features_EURUSD_X']
            for table in feature_tables:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table} (date)")
                self.quality_report['actions'].append(f"Created index idx_{table}_date")
                
            # Create indexes for transaction and aggregate tables
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions (Time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions (Account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_account_aggregates_account ON features_account_aggregates (Account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_daily_aggregates_date ON features_daily_aggregates (date)")
            self.quality_report['actions'].append("Created indexes for transactions and aggregate tables")
            
            # Create partitioned-like structure for transactions by year
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions_2023 (
                    Time DATETIME,
                    Account TEXT,
                    Amount REAL,
                    Class INTEGER,
                    V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL,
                    V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL,
                    V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL,
                    CHECK (Time >= '2023-01-01 00:00:00' AND Time < '2024-01-01 00:00:00')
                )
            """)
            cursor.execute("""
                INSERT INTO transactions_2023
                SELECT * FROM transactions
                WHERE Time >= '2023-01-01 00:00:00' AND Time < '2024-01-01 00:00:00'
            """)
            self.quality_report['actions'].append("Created and populated transactions_2023 partition")
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error optimizing database: {str(e)}")
            self.quality_report['issues'].append(f"Database optimization error: {str(e)}")

    def store_data(self, df: pd.DataFrame, table_name: str):
        """Store feature data in SQLite database with verification"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=True)
            # Verify storage
            stored_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", conn)['count'].iloc[0]
            conn.close()
            if stored_count != len(df):
                self.logger.warning(f"Storage verification failed for {table_name}: expected {len(df)} rows, got {stored_count}")
                self.quality_report['issues'].append(f"Storage verification failed for {table_name}: {len(df)} vs {stored_count}")
            else:
                self.logger.info(f"Successfully stored {stored_count} rows in {table_name}")
                self.quality_report['actions'].append(f"Stored {stored_count} rows in table {table_name}")
        except Exception as e:
            self.logger.error(f"Error storing data in {table_name}: {str(e)}")
            self.quality_report['issues'].append(f"Storage error for {table_name}: {str(e)}")

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save feature data to CSV in output folder"""
        try:
            output_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(output_path)
            self.logger.info(f"Saved feature data to {output_path}")
            self.quality_report['actions'].append(f"Saved feature data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving to {output_path}: {str(e)}")
            self.quality_report['issues'].append(f"CSV save error for {output_path}: {str(e)}")

    def generate_feature_report(self, output_path: str = os.path.join(OUTPUT_DIR, "feature_engineering_report.json")):
        """Generate and save feature engineering report"""
        try:
            with open(output_path, "w") as f:
                json.dump(self.quality_report, f, indent=2, default=str)
            self.logger.info(f"Generated feature report at {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating feature report: {str(e)}")
            self.quality_report['issues'].append(f"Feature report error: {str(e)}")

def main():
    engineer = FeatureEngineer()
    
    # Process market data
    symbols = ["^GSPC", "EURUSD=X"]
    for symbol in symbols:
        table_name = f"market_{symbol.replace('=', '_')}"
        try:
            conn = sqlite3.connect(engineer.db_path)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, index_col='date', parse_dates=['date'])
            conn.close()
            
            # Compute technical indicators
            feature_df = engineer.compute_technical_indicators(df, symbol)
            engineer.store_data(feature_df, f"features_{symbol.replace('=', '_')}")
            engineer.save_to_csv(feature_df, f"features_{symbol.replace('=', '_')}.csv")
        except Exception as e:
            engineer.logger.error(f"Error processing market data for {symbol}: {str(e)}")
            engineer.quality_report['issues'].append(f"Market data processing error for {symbol}: {str(e)}")
    
    # Process transaction data
    try:
        conn = sqlite3.connect(engineer.db_path)
        fraud_df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=['Time'])
        conn.close()
        
        # Aggregate transactions
        account_agg, daily_agg = engineer.aggregate_transactions(fraud_df)
        if not account_agg.empty:
            engineer.store_data(account_agg, "features_account_aggregates")
            engineer.save_to_csv(account_agg, "features_account_aggregates.csv")
        if not daily_agg.empty:
            engineer.store_data(daily_agg, "features_daily_aggregates")
            engineer.save_to_csv(daily_agg, "features_daily_aggregates.csv")
    except Exception as e:
        engineer.logger.error(f"Error processing transaction data: {str(e)}")
        engineer.quality_report['issues'].append(f"Transaction data processing error: {str(e)}")
    
    # Optimize database schema
    engineer.optimize_database()
    
    # Generate feature report
    engineer.generate_feature_report()

if __name__ == "__main__":
    main()