# Nexus Bank Analytics Pipeline

## Project Overview

As Nexus Bank's analytics team, we are developing a comprehensive data pipeline to collect, store, and analyze financial datasets. This project focuses on enabling accurate portfolio forecasting and real-time fraud detection through robust data engineering and analysis.

## Phase 1: Data Foundation (Week 1)

### Objectives

Establish a strong data foundation by sourcing, retrieving, and analyzing relevant financial datasets to support portfolio forecasting and fraud detection capabilities.

### Datasets

We are working with the following key datasets:

- **S&P 500 Stock Prices**: Historical and real-time market data for portfolio analysis
- **Cryptocurrency Data**: Digital asset prices and trading volumes
- **Macroeconomic Indicators**: Federal Reserve economic data (interest rates, inflation, GDP)
- **Credit Card Fraud Dataset**: Transaction data for fraud detection model training

## Getting Started

### Prerequisites

```bash
pip install pandas numpy requests yfinance fredapi matplotlib seaborn jupyter
```

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/nexus-bank/analytics-pipeline.git
cd analytics-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (create `.env` file):
```
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
```

## Data Sources & Retrieval Methods

### S&P 500 Data
- **Source**: Yahoo Finance via `yfinance` Python library
- **Endpoint**: `yf.download()` for historical data
- **Update Frequency**: Daily
- **Retrieval Method**: Automated API calls

### Cryptocurrency Data
- **Source**: CoinGecko API
- **Endpoint**: `https://api.coingecko.com/api/v3/`
- **Coverage**: Top 50 cryptocurrencies by market cap
- **Update Frequency**: Hourly

### Macroeconomic Indicators
- **Source**: Federal Reserve Economic Data (FRED)
- **API**: FRED API via `fredapi` library
- **Key Indicators**: GDP, unemployment rate, federal funds rate, CPI
- **Update Frequency**: Monthly/Quarterly (varies by indicator)

### Fraud Detection Dataset
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Method**: Manual download and preprocessing
- **Size**: ~284,000 transactions with fraud labels
- **Features**: Anonymized transaction features (V1-V28) + Amount, Time, Class

## Project Structure

```
nexus-bank-analytics/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Cleaned and transformed data
│   └── external/               # External reference data
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_summary_statistics.ipynb
├── src/
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   ├── crypto_data.py
│   │   ├── macro_indicators.py
│   │   └── fraud_data.py
│   └── utils/
│       ├── __init__.py
│       └── summary_stats.py
├── logs/
│   ├── acquisition_log.md
│   └── ai_usage_log.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Summary Statistics Generated

For each dataset, we produce comprehensive summary statistics including:

- **Descriptive Statistics**: Mean, median, mode, standard deviation, variance
- **Distribution Metrics**: Skewness, kurtosis, percentiles (25th, 50th, 75th, 95th)
- **Data Quality Metrics**: Missing values count/percentage, duplicate records
- **Temporal Analysis**: Data range, frequency, gaps in time series
- **Correlation Analysis**: Cross-correlation between key variables

## AI Tools Integration

This project leverages AI tools to enhance productivity and code quality:

### GitHub Copilot
- **Usage**: Code completion and function generation
- **Application**: Data retrieval scripts, statistical analysis functions
- **Logged Activities**: Function suggestions, debugging assistance

### ChatGPT/Claude
- **Usage**: Documentation generation, code optimization suggestions
- **Application**: README creation, code commenting, error resolution
- **Logged Activities**: Query topics, implementation suggestions used

All AI tool usage is documented in `logs/ai_usage_log.md` for transparency and evaluation.

## Data Acquisition Log

Detailed acquisition information is maintained in `logs/acquisition_log.md` including:

- Data source URLs and API endpoints
- Retrieval timestamps and methods
- Data quality checks performed
- Any issues encountered and resolutions
- File sizes and record counts

## Usage Examples

### Quick Data Summary
```python
from src.utils.summary_stats import generate_summary
from src.data_acquisition.market_data import get_sp500_data

# Retrieve S&P 500 data
sp500_data = get_sp500_data(start_date='2020-01-01', end_date='2024-01-01')

# Generate summary statistics
summary = generate_summary(sp500_data)
print(summary)
```

### Running All Data Acquisition
```python
# Execute complete data pipeline
python src/main.py --mode acquisition --datasets all
```

## Evaluation Criteria Compliance

This project meets capstone evaluation standards through:

- **Data Engineering Quality**: Robust error handling, logging, and data validation
- **Communication Clarity**: Comprehensive documentation and code comments
- **Technical Proficiency**: Efficient use of APIs, proper data structures
- **AI Integration**: Strategic use of AI tools with proper documentation

## Next Steps (Upcoming Weeks)

- **Week 2**: Data cleaning, transformation, and storage optimization
- **Week 3**: Exploratory data analysis and feature engineering
- **Week 4**: Model development for portfolio forecasting
- **Week 5**: Fraud detection algorithm implementation
- **Week 6**: Pipeline integration and performance optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-dataset`)
3. Commit changes (`git commit -am 'Add new dataset integration'`)
4. Push to branch (`git push origin feature/new-dataset`)
5. Create Pull Request

## License

This project is proprietary to Nexus Bank. All rights reserved.

## Contact

**Nexus Bank Analytics Team**
- Email: analytics@nexusbank.com
- Slack: #analytics-pipeline
- Project Lead: [Your Name]

## Acknowledgments

- Federal Reserve Economic Data (FRED) for macroeconomic indicators
- Yahoo Finance for market data access
- CoinGecko for cryptocurrency data
- Kaggle community for fraud detection dataset

---

*Last Updated: June 2025*
*Project Status: Phase 1 - Data Foundation*
