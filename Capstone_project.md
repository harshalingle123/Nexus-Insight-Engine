
Financial Analytics and Modeling Project for Nexus Bank

Introduction

Nexus Bank invites proposals for a six-week capstone in financial data engineering. In the fast-moving finance sector, robust data pipelines are crucial for tasks like algorithmic trading, risk modeling, and fraud detection. Participants will work in teams to process large financial datasets, simulating a role in the bank’s analytics division. This hands-on project will combine data ingestion, transformation, and analysis to derive actionable insights from market and transaction data.

Project Scope and Objectives

The scenario: Nexus Bank wants to leverage data for portfolio analysis and fraud prevention. Key objectives include:
Data Sourcing: Identify public financial datasets: e.g., historical stock prices (via Yahoo Finance API), cryptocurrency data (Kaggle), and macroeconomic indicators (Federal Reserve Economic Data). For transactional analysis, use datasets like the Kaggle credit card fraud dataset.

Data Cleaning: Handle missing data during market holidays, remove outliers or erroneous trades, and standardize formats (timestamps, currency conversions). Document cleaning rationale.

Data Ingestion & Storage: Build pipelines to fetch and store streaming market data (stock prices, FX rates) and batch data (daily transaction logs) into a database. Implement techniques like partitioning (e.g., by date or asset) to improve query speed.

Data Transformation: Engineer financial features (e.g., moving averages, volatility indicators, RSI) and aggregate transaction data into summary metrics (e.g., total volume per account). Test calculations against benchmarks or known formulas.

Modeling & Analysis: Develop use cases: (a) Time-series forecasting of asset prices; (b) Classification model for fraud detection. Perform backtesting or validation and interpret the results.
Reporting & Visualization: Create dashboards (e.g., price over time, portfolio risk heatmaps, fraud incident timelines) to support decision making.


Students will tackle realistic problems such as aligning data from multiple markets (different trading hours) and ensuring data integrity under shifting market conditions. By using authoritative sources like Yahoo Finance and FRED, teams will gain experience typical of quantitative finance and risk management roles.

Phases and Weekly Deliverables

The six-week timeline is divided as follows:

Phase 1 (Weeks 1–2): Data Collection & Preprocessing

Week 1: Identify and retrieve relevant datasets (e.g., historical S&P 500 prices, exchange rates, economic indicators). Document API usage or data endpoints. Deliverable: Data acquisition log and summary statistics of raw data.

Week 2: Clean and preprocess data: handle missing values (e.g., fill with interpolation), remove anomalies, and normalize data units (e.g., currency). Deliverable: Cleaned datasets and data quality report.

Phase 2 (Weeks 3–4): Pipeline Development & Feature Engineering

Week 3: Develop automated data ingestion pipelines (e.g., Python scripts or scheduled jobs) to populate a database or data warehouse. Include real-time or periodic updates. Deliverable: Working ingestion pipeline with logs.

Week 4: Engineer analysis-ready features: compute indicators (moving averages, Bollinger bands, etc.) and aggregate transactional data by account or day. Optimize the database schema (indexes, partitions by date/asset). Deliverable: Feature tables and documentation of transformation logic.

Phase 3 (Weeks 5–6): Modeling, Visualization, & Reporting

Week 5: Perform analytics: build predictive models (e.g., ARIMA or ML for price forecasting; logistic regression or ensemble for fraud detection).
Backtest  on  historical data and evaluate performance. Deliverable:
Modeling code and performance evaluation report.

Week 6: Create visual dashboards (e.g., price trends, risk heatmaps, ROC curves for fraud detection). Finalize a comprehensive report and presentation. Deliverable: Interactive visualizations and project summary presentation.


Weekly check-ins with mentors will help students verify financial assumptions and correctness (e.g., ensuring no look-ahead bias). Teams should collaborate; roles may include market data specialist, pipeline engineer, modeler, and report writer.

Use of Generative AI

Students should leverage AI tools to enhance productivity and insight:

ChatGPT (OpenAI): Use ChatGPT for generating boilerplate code (e.g., data fetching scripts), explaining financial concepts (e.g., what an indicator measures), or suggesting approaches for feature engineering. ChatGPT can help outline how to compute complex financial ratios.

GitHub Copilot: Assist coding in Python or SQL, especially for repetitive tasks like writing loops over time-series data or formulating complex queries (e.g., generating cross-correlation matrices). Copilot accelerates routine coding.

Gemini (Google): Employ Gemini to retrieve current examples of financial analyses (e.g., code snippets for time-series models) or to suggest libraries and techniques for market data. Gemini’s access to up-to-date information can guide method selection.

Claude (Anthropic): Use Claude for summarizing financial news or reports to contextualize data. For example, Claude can produce concise summaries of quarterly earnings that might explain stock movements.

LangChain: Apply LangChain to build chains that combine data queries with LLMs. For instance, feed model predictions into a prompt that asks for a business interpretation, or iterate multiple prompts for complex analysis.
Teams should document AI usage (e.g., note if GPT-4 generated a key function or if Claude improved report clarity).

Expected Outcomes

Upon completion, teams will:

Demonstrate robust ingestion and warehousing of financial and transactional data.

Produce	cleaned,	integrated	datasets	combining	market	prices	and macroeconomic indicators.

Perform thorough EDA and develop predictive models for financial forecasting or fraud detection.

Present findings via dashboards that illustrate data-driven financial insights (e.g., risk metrics, trend analyses).

Reflect on how AI tools streamlined the workflow and contributed insights.


This capstone prepares students for finance-focused data engineering roles, reinforcing skills in handling time-series data and financial modeling.

Deliverables

A GitHub repository with all code (notebooks, scripts) and documentation.

Cleaned and transformed datasets with descriptions of each data field.

Phase-wise progress reports, including analysis results and model evaluations.

Predictive model outputs with evaluation metrics (e.g., forecast accuracy, fraud detection AUC).

Interactive dashboards or visual reports (e.g., using Tableau or Python libraries) demonstrating key finance insights.

Final presentation summarizing technical approach and business implications.
Proposal Submission Requirements

Submit a PDF containing:

Project Overview: Goals (e.g., accuracy targets for forecasting, fraud detection objectives).

Data Strategy: List planned datasets (Yahoo Finance, FRED, etc.) and approach to accessing them.

Technical Plan: Tools/languages (e.g., Python, SQL, database systems) and pipeline architecture.

Team and Timeline: Roles and week-by-week schedule (1–6).

Experience: Relevant skills (e.g., past work with financial data or ML).


Proposals should avoid proprietary data; use only approved public sources.

Evaluation Criteria

Teams will be evaluated on:

Data Engineering Quality: Efficiency and scalability of data pipelines and storage design.

Analytical Rigor: Accuracy of models, thoroughness of EDA, and correctness of financial calculations.

AI Integration: Thoughtful use of AI tools (e.g., ChatGPT for insight vs. copying answers).

Insight Generation: Business relevance of insights (e.g., improved portfolio strategies, fraud detection effectiveness).

Communication: Clarity of documentation, visualizations, and presentations in a financial context.

Submission Deadline

The Project is due by July 10, 2025 (Tentative) via email.