# Dynamic Portfolio Backtesting with Random Forest (Shiny App)

📊 This repository hosts an interactive **R Shiny application** developed for the course *Machine Learning in Factor Investing* (MSc Finance – emlyon business school).  
The project implements a **dynamic portfolio backtesting strategy** based on **Random Forest models** applied to US stock and macroeconomic data (1991–2025).  

---

## ✨ Project Overview

The objective of this project was to design and backtest a **time-varying investment strategy** using machine learning signals.  
Unlike static allocations, the portfolio is **re-estimated monthly** based on new predictions of forward returns.

Main choices:
- **Dataset:** ~500 US firms (with financial + ESG features) and 4 macro indicators (`spx_macro.RData` provided in the course).
- **Model:** Random Forest (`ranger` package) predicting one-month forward returns (`fwd_return`).
- **Features:** momentum (1M, 3M), volatility trends, valuation ratios (P/B), leverage (D/E), profitability, size, etc.
- **Portfolio construction:**  
  - Re-train model monthly with rolling window.  
  - Rank stocks by predicted forward returns.  
  - Select **top 20%** for the portfolio.  
  - Assign **weights proportional to predicted returns** (not equal-weighted).  
- **Benchmark:** equal-weighted portfolio of all stocks.

The project is deployed as a **Shiny web dashboard** where the user can:
- Select backtest period and training start date.  
- Adjust the number of Random Forest trees.  
- Launch backtests interactively and visualize performance metrics.  

---

## 🚀 Live Demo
👉 https://jawad-kassimi.shinyapps.io/Machine_Learning_Project/
---

## 🖥️ Running the App Locally

Clone the repository and open the project in RStudio, or run directly in R:

```r
# Install required packages (if not already installed)
install.packages(c("shiny", "shinydashboard", "shinyWidgets", 
                   "plotly", "DT", "tidyverse", "ranger", "zoo", "MASS"))

# Run the app
shiny::runApp("path/to/project")
