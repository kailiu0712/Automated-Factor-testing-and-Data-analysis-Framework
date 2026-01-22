# Factor Test Code 2.0  
**A-Share Equity Factor Backtesting & Evaluation Pipeline**

A modular factor-testing framework for China A-share equities that supports **data extraction**, **local caching**, **rank IC / ICIR evaluation**, **decile portfolio return analysis**, and **automatic PDF report generation**.  
The system is optimized for **batch execution** to avoid memory overflow and supports **incremental updates** for recent data.

---

## 📁 Project Structure

This repository consists of **four core Python modules**:
```
├── main_2.0.py
├── data_storage_new.py
├── calculate_icir_new.py
└── calculate_returns_new.py
```
### Module Overview

- **`main_2.0.py`**  
  Entry point. Controls execution flow and configuration.  
  - Data extraction / update  
  - Factor IC/ICIR testing  
  - Return calculation and report generation  

- **`data_storage_new.py`**  
  - Connects to SQL Server via `pymssql`
  - Extracts factor tables, base columns, and stock pool indicators
  - Saves data locally in CSV / Parquet format
  - Supports **incremental update mode**

- **`calculate_icir_new.py`**  
  - Computes cross-sectional **Rank IC**, **IC Std**, and **ICIR**
  - Supports multiple return definitions and stock pools
  - Optional abnormal-regime and month-state filtering
  - Batch processing for memory safety

- **`calculate_returns_new.py`**  
  - Computes **decile (quantile) portfolio returns**
  - Generates bar plots for factor returns
  - Creates factor-level PDF reports
  - Merges all PDFs into a single final report

---

## ✨ Key Features

- End-to-end factor testing pipeline
- Batch processing to avoid memory overflow
- Incremental update for recent years
- Multiple stock universes supported
- Automatic PDF report generation
- Fully configurable testing framework

---

## 🧰 Requirements

### Python Dependencies

```bash
pip install pandas numpy pymssql tqdm numba matplotlib reportlab PyPDF2

