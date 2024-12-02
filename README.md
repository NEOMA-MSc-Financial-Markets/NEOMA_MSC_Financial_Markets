# NEOMA_MSC_Financial_Markets

<img width="967" alt="Screenshot 2024-12-02 at 18 10 47" src="https://github.com/user-attachments/assets/063a1e99-7878-41d5-b6a0-8e880ab636ad">

This Python project implements a comprehensive portfolio optimization tool for a 5-stock portfolio. It fetches historical data, calculates key metrics like GMVP and MSR, and visualizes results. The script incorporates advanced concepts such as Fama-French 3-factor model, efficient frontier, and SML analysis, making it suitable for asset-management.

NEOMA World Growth Portfolio (NWGP) Optimization Tool

Project Overview

This Python project implements a sophisticated portfolio optimization and analysis tool for the NEOMA World Growth Portfolio (NWGP). The NWGP is designed to achieve optimal returns while managing risks in the face of global economic and geopolitical uncertainties. It focuses on undervalued opportunities in the automotive, energy, financial, and defense sectors, diversifying across various global markets including the U.S., India, Europe, and Japan.

Objective

The primary objective of this tool is to analyze and optimize a portfolio of 5 carefully selected stocks:

	▪	Michelin (ML.PA) - Automotive sector, Tires
	▪	HDFC Bank (HDB) - Financial Services (Banking) from India
	▪	Constellation Energy (CEG) - Energy, Nuclear Power
	▪	Raytheon Technologies (RTX) - Defense sector
	▪	Mitsui & Co. (8031.T) - Conglomerate, Japan

The tool aims to maximize returns while minimizing risks, maintaining a minimum allocation of 5% per asset to ensure exposure to all selected sectors.

Code Overview

This Python project implements a sophisticated portfolio optimization and analysis tool for financial markets. It provides comprehensive functionality for analyzing a set of stocks, computing various portfolio metrics, and visualizing results through multiple plots and charts.

Key Features

	•	Data retrieval from Yahoo Finance API
	•	Portfolio optimization (Global Minimum Variance Portfolio and Maximum Sharpe Ratio)
	•	Efficient frontier calculation and visualization
	•	Monte Carlo simulation for portfolio analysis
	•	Fama-French 3-factor model analysis
	•	Security Market Line (SML) visualization
	•	Performance evaluation against a market benchmark
	•	Interactive data filtering for extreme values

Requirements

	•	Python 3.11 (This code won't work in python3.12 env)
	•	Dependencies: yfinance, numpy, pandas, scipy, matplotlib, seaborn, statsmodels
Usage

	1.	Configure TICKERS, START_DATE, END_DATE, and RISK_FREE_RATE constants in the script.
	2.	Execute the script: python portfolio_analysis.py
	3.	Follow prompts to include/exclude stocks with extreme values.

Running the Project

To simplify the setup process and ensure all dependencies are correctly installed, we've provided a bash script that creates a virtual environment, installs the required packages, and runs the main Python script. Follow these steps to use it:
Make sure you have Python 3.11 installed on your system.

Save the following script as run_portfolio_optimization.sh in the same directory as your main_portfolio_optimization.py file:
bash

	#!/bin/bash

	# Create a virtual environment
	python3.11 -m venv portfolio_env

	# Activate the virtual environment
	source portfolio_env/bin/activate

	# Upgrade pip
	pip install --upgrade pip

	# Install required libraries
	pip install yfinance numpy pandas scipy matplotlib seaborn statsmodels pandas-datareader

	# Run the Python script
	python3.11 main_portfolio_optimization.py

	# Deactivate the virtual environment
	deactivate

Make the script executable by running:

	chmod +x run_portfolio_optimization.sh

Run the script:

	./run_portfolio_optimization.sh

This script will create a virtual environment, install all necessary dependencies, run the main Python script, and then deactivate the virtual environment.

Note: If you're using Windows, you'll need to modify the script or use a bash emulator like Git Bash to run this script.

Core Functions

	•	compute_stock_metrics(): Calculates returns, volatilities, and correlation matrix
	•	optimize_portfolio(): Computes GMVP and MSR portfolios
	•	calculate_efficient_frontier(): Generates efficient frontier points
	•	simulate_portfolios(): Performs Monte Carlo simulation
	•	plot_historical_prices(), plot_correlation_heatmap(), plot_gmvp_msr_weights(), plot_return_vs_risk_portfolio_GMVP_MSR(): Visualization functions
	•	compute_fama_french_3_factors(): Implements Fama-French 3-factor model analysis
	•	plot_sml_with_dynamic_gmvp(): Plots Security Market Line with GMVP
	•	evaluate_gmvp_performance(): Evaluates GMVP performance against a benchmark

Output

Printing in terminal : 

<img width="653" alt="Screenshot 2024-12-02 at 19 17 56" src="https://github.com/user-attachments/assets/d109d002-80c4-4813-8358-4dac93f3bd80">
<img width="523" alt="Screenshot 2024-12-02 at 19 18 17" src="https://github.com/user-attachments/assets/82b76d56-53a8-4f11-b240-06c24f47b052">



The script generates a series of plots and prints detailed portfolio metrics, including:

	•	Historical price charts
	•	Correlation heatmaps
	•	Portfolio weight distributions
	•	Risk-return scatter plots with efficient frontier
	•	Fama-French factor analysis results
	•	Security Market Line analysis

Risk-return scatter plots with efficient frontier with GMVP (GLOBAL MINIMUM VARIANCE PORTFOLIO) & (MSR MAXIMUM SHARPE RATIO)

![Exp Return-Volatitlities](https://github.com/user-attachments/assets/47f44489-b34b-41bb-be2a-882a2d12abc9)


Portfolio weight distributions

![Portfolio_weight_distributions](https://github.com/user-attachments/assets/8f2c3a7b-2eec-46c2-9ea2-73b0a0cc0e3c)

Heatmap Correlation Matrix

![Figure 2024-12-02 191338](https://github.com/user-attachments/assets/1711b5e0-981f-40af-ab97-ae036b1f2393)


Note
This report, created by NEOMA Business School students for the Financial Data Analytics & Programming course, is for educational and research purposes in the context of advanced financial analysis and portfolio management purposes only. It is hypothetical and does not constitute financial or investment advice or an invitation to invest.

AUTHOR : Adrien PICARD - picard.adrien@icloud.com
