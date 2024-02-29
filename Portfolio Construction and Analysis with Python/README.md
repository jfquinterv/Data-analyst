# Portfolio Repository for Data Analysis Projects

## Introduction

This repository showcases my data analysis projects, demonstrating my skills in financial data analysis and portfolio construction. These projects are part of my portfolio for potential employers and educational purposes, highlighting my ability to apply theoretical knowledge to practical, real-world data.

## Projects Overview

### 1. Quantitative Analysis Report (`Informe Cuantitativo`)

This project applies concepts learned from the [Investment Management with Python and Machine Learning Specialization](https://www.coursera.org/learn/introduction-portfolio-construction-python?specialization=investment-management-python-machine-learning) on Coursera to analyze real financial data. Using the `yfinance` API, the project fetches historical price data for various assets to perform risk and return analysis, portfolio optimization, and more. This approach allows for the application of theoretical concepts learned in the course with up-to-date data from the stock market, demonstrating the practical application of investment strategies with real-world financial data. Detailed documentation can be found in the project's directory.


## Project Overview

The project consists of several Jupyter notebooks and Python scripts that demonstrate various data analysis and investment management concepts, including:

- Risk and return analysis
- Portfolio construction
- Performance measurement
- Analysis of industry-specific returns and hedge fund indices

## Files Description

- `Informe Cuantitativo.ipynb`: Quantitative analysis report, including detailed data analysis and visualization.
- `Week 1.ipynb`, `Week 2.ipynb`: Weekly progress notebooks containing exercises and explorations based on the Coursera course.
- `risk.py`: Python script for risk assessment functions.
- `ind30_m_vw_rets.csv`, `Portfolios_Formed_on_ME_monthly_EW.csv`, `edhec-hedgefundindices.csv`: Datasets used for analysis.

## Technologies and Data Sources

- **Python**: Used for data analysis and modeling, including libraries like `numpy`, `pandas`, `matplotlib`, `scipy`, and `yfinance`.
- **Yahoo Finance API (`yfinance`)**: Provides real financial data for the "Informe Cuantitativo" project, enabling analysis of current market trends and asset performance. The use of `yfinance` is crucial for accessing up-to-date and historical stock price information, which forms the basis for all analyses conducted in the project.

## Installation

Before running the projects, you need to install the necessary Python libraries, including `yfinance` for fetching financial data. You can install all required libraries using the following command:

```bash
pip install numpy pandas matplotlib scipy yfinance
