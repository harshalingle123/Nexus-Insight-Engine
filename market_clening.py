"""
Nexus Bank Financial Analytics Project - Week 2 Data Cleaning
Author: Nexus Bank Analytics Team
Date: June 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import logging
import requests
import os
import json

# Create directories if they don't exist
LOG_DIR = "logs"
OUTPUT_DIR = "output"
DATASET_DIR = "dataset"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'nexus_bank_cleaning.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Class to clean and preprocess financial datasets"""
    
    def __init__(self, db_path: str = os.path.join(OUTPUT_DIR, "nexus_bank.db")):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.DataCleaner")
        self.quality_report = {"issues": [], "actions": [], "metrics": {}}

    def fetch_from_api(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data using the Yahoo Finance API (from market_data.py)"""
        try:
            self.logger.info(f"Fetching data for {symbol}")
            # Simulate API call using market_data.py's YahooFinanceAPI
            url = f"http://localhost:5000/api/stock/{symbol}?period={period}"
            response = requests.get(url)
            data = response.json()
            df = pd.DataFrame(data["historical_data"])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            self.quality_report["issues"].append(f"Fetch error for {symbol}: {str(e)}")
            return pd.DataFrame()

    def clean_market_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean market data (stocks, forex, indices)"""
        original_rows = len(df)
        self.logger.info(f"Cleaning data for {symbol} with {original_rows} rows")

        # Handle missing values (e.g., market holidays)
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            self.quality_report["issues"].append(f"Found {missing_count} missing values in {symbol}")
            df = df.interpolate(method="linear", limit_direction="both")  # Interpolate
            df = df.fillna(method="ffill").fillna(method="bfill")  # Forward/backward fill
            self.quality_report["actions"].append(f"Interpolated and filled {missing_count} missing values")

        # Remove outliers (e.g., prices > 5 std devs from mean)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = (df[col] > mean + 5 * std) | (df[col] < mean - 5 * std)
            outlier_count = outliers.sum()
            if outlier_count > 0:
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate()
                self.quality_report["issues"].append(f"Found {outlier_count} outliers in {symbol}.{col}")
                self.quality_report["actions"].append(f"Removed {outlier_count} outliers in {col}")

        # Normalize units (ensure prices are float, volume is int)
        for col in price_cols:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)

        # Remove duplicate dates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            df = df[~df.index.duplicated(keep="first")]
            self.quality_report["issues"].append(f"Found {duplicates} duplicate dates in {symbol}")
            self.quality_report["actions"].append(f"Removed {duplicates} duplicate dates")

        cleaned_rows = len(df)
        self.quality_report["metrics"][symbol] = {
            "original_rows": int(original_rows),  # Convert to Python int
            "cleaned_rows": int(cleaned_rows),    # Convert to Python int
            "missing_values_handled": int(missing_count),  # Convert to Python int
            "outliers_removed": int(outlier_count),       # Convert to Python int
            "duplicates_removed": int(duplicates)         # Convert to Python int
        }
        return df

    def clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Kaggle credit card fraud dataset"""
        original_rows = len(df)
        self.logger.info(f"Cleaning transaction data with {original_rows} rows")

        # Handle missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            df = df.dropna()  # Drop rows with missing values (fraud data is sensitive)
            self.quality_report["issues"].append(f"Found {missing_count} missing values in transaction data")
            self.quality_report["actions"].append(f"Dropped {missing_count} rows with missing values")

        # Remove outliers in 'Amount' (e.g., > 5 std devs)
        mean = df["Amount"].mean()
        std = df["Amount"].std()
        outliers = (df["Amount"] > mean + 5 * std) | (df["Amount"] < 0)  # Negative amounts are invalid
        outlier_count = outliers.sum()
        if outlier_count > 0:
            df = df[~outliers]
            self.quality_report["issues"].append(f"Found {outlier_count} outliers in transaction Amount")
            self.quality_report["actions"].append(f"Removed {outlier_count} outlier transactions")

        # Standardize formats, explicitly handling dtype for Time column
        df = df.copy()  # Create a copy to avoid modifying a slice
        df["Time"] = pd.to_datetime(df["Time"], unit="s")  # Convert Time to datetime
        df.loc[:, "Amount"] = df["Amount"].astype(float)
        df.loc[:, "Class"] = df["Class"].astype(int)  # Fraud label (0 or 1)

        cleaned_rows = len(df)
        self.quality_report["metrics"]["transactions"] = {
            "original_rows": int(original_rows),  # Convert to Python int
            "cleaned_rows": int(cleaned_rows),    # Convert to Python int
            "missing_values_handled": int(missing_count),  # Convert to Python int
            "outliers_removed": int(outlier_count)        # Convert to Python int
        }
        return df

    def store_data(self, df: pd.DataFrame, table_name: str):
        """Store cleaned data in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=True)
            conn.close()
            self.logger.info(f"Stored cleaned data in {table_name}")
            self.quality_report["actions"].append(f"Stored cleaned data in table {table_name}")
        except Exception as e:
            self.logger.error(f"Error storing data in {table_name}: {str(e)}")
            self.quality_report["issues"].append(f"Storage error for {table_name}: {str(e)}")

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save cleaned data to CSV in output folder"""
        try:
            output_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(output_path)
            self.logger.info(f"Saved cleaned data to {output_path}")
            self.quality_report["actions"].append(f"Saved cleaned data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving to {output_path}: {str(e)}")
            self.quality_report["issues"].append(f"CSV save error for {output_path}: {str(e)}")

    def generate_quality_report(self, output_path: str = os.path.join(OUTPUT_DIR, "data_quality_report.json")):
        """Generate and save data quality report"""
        try:
            # Convert any NumPy types to Python types in quality_report
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            serializable_report = convert_numpy_types(self.quality_report)
            with open(output_path, "w") as f:
                json.dump(serializable_report, f, indent=2)
            self.logger.info(f"Generated data quality report at {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating quality report: {str(e)}")
            print("generate_quality_report error generated")

def main():
    cleaner = DataCleaner()

    # Fetch and clean market data
    symbols = ["^GSPC", "EURUSD=X"]
    for symbol in symbols:
        df = cleaner.fetch_from_api(symbol, period="1y")
        if not df.empty:
            cleaned_df = cleaner.clean_market_data(df, symbol)
            cleaner.store_data(cleaned_df, f"market_{symbol.replace('=', '_')}")
            cleaner.save_to_csv(cleaned_df, f"cleaned_{symbol.replace('=', '_')}.csv")

    # Clean transaction data (assuming Kaggle dataset is in dataset folder)
    creditcard_path = os.path.join(DATASET_DIR, "creditcard.csv")
    if os.path.exists(creditcard_path):
        fraud_df = pd.read_csv(creditcard_path)
        cleaned_fraud_df = cleaner.clean_transaction_data(fraud_df)
        cleaner.store_data(cleaned_fraud_df, "transactions")
        cleaner.save_to_csv(cleaned_fraud_df, "cleaned_transactions.csv")
    else:
        cleaner.quality_report["issues"].append(f"Credit card fraud dataset not found at {creditcard_path}")
        cleaner.logger.warning(f"Credit card fraud dataset not found at {creditcard_path}")

    # Generate quality report
    cleaner.generate_quality_report()

if __name__ == "__main__":
    main()