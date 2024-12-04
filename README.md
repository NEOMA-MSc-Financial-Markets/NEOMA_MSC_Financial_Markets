# NEOMA_MSC_Financial_Markets

<img width="967" alt="Screenshot 2024-12-02 at 18 10 47" src="https://github.com/user-attachments/assets/063a1e99-7878-41d5-b6a0-8e880ab636ad">

Collaborators : 

Adrien PICARD 					Constance BERRAUTE				Louis DELACOUR		
adrien.picard.23@neoma-bs.com			constance.berraute.21@neoma-bs.com		louis.delacour.21@neoma-bs.com
MSc Financial Markets & Risk Management		MSc Financial Markets & Risk Management		MSc Financial Markets & Risk Management

Lucas MOERLEN					Sacha DIDUCH
lucas.moerlen.21@neoma-bs.com			sacha.diduch.24@neoma-bs.com
MSc Financial Markets & Risk Management		MSc Financial Markets & Risk Management


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

	•	Python 3.11 or newer
	•	Dependencies: yfinance, numpy, pandas, scipy, matplotlib, seaborn, statsmodels (requirements.txt)

Installation and Setup

	Clone the repository or download the script.

Create a virtual environment:

	python3 -m venv portfolio_env

Activate the virtual environment:
	
 	On Windows: portfolio_env\Scripts\activate
	On macOS and Linux: source portfolio_env/bin/activate

Install required packages:

	pip install yfinance numpy pandas scipy matplotlib seaborn statsmodels pandas_datareader

Usage

Configure the portfolio parameters in the config.json file:
	
 	tickers: List of stock tickers
	start_date: Start date for historical data
	end_date: End date for historical data (optional, defaults to current date)
	risk_free_rate: Risk-free rate for calculations
	output_path: Directory for saving output plots
	base_currency: Base currency for conversions

Run the script:

	python portfolio_optimization.py

Follow any prompts to include/exclude stocks with extreme values.

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


Customization
	
  	You can easily customize the portfolio by modifying the config.json file. Add or remove tickers, adjust dates, or change the risk-free rate to suit your analysis needs.

Note : This tool is designed for educational and research purposes in the context of advanced financial analysis and portfolio management. It does not constitute financial or investment advice.

Author
	
 	Adrien PICARD - picard.adrien@icloud.com

License : This project is open-source and available under the MIT License.
