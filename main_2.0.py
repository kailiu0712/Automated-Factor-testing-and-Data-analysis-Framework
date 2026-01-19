# main_2.0.py
import os
import time
from data_storage_new import run_data_storage
from calculate_icir_new import run_icir_calculation
from calculate_returns_new import run_return_calculation

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Factor Test Code 2.0
# 1. Run the main file to test.
# 2. Function: Extract data from the database and save it locally, draw the grouped return rate chart, calculate the IC and ICIR index table, and automatically generate the factor report (factor names are arranged in alphabetical order).
# 3. Multi-process is cancelled for now and batch operation is added to avoid memory overflow. The full-time and full-factor re-extraction takes about 1 hour, and the test takes about 30 minutes.
# 4. The stock pool data starts from 2013, the regular factor data starts from 2010, and the risk factor data starts from 2013.
# 5. Efficiency optimization direction: Change to a faster reading temporary file type such as parquet; add multi-process without affecting memory.
# 6. Currently, the component stock indicators use the StockIndexWeightNew2 table and do not use the adjusted version. The stock pool indicators use the adjusted version.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Parameter Settings
    # Set the start and end intervals for data extraction or testing
    start_date = '2014-01-01'
    end_date = '2024-12-18'
    # Path for storing factor values
    factor_dir = r'D:\code\results\All_stocks_with_stockpool3.0'
    # Test result output path
    output_dir = r'D:\code\results\All_stocks_with_stockpool3.0\output'
    # The major category parameters of the factors to be stored or tested. Please note that when storing, do not include "Risk&Div", and it will still be stored by default
    name_list = ['ValuationNew', 'ValuationRankNew', 'Technical1New', 'Technical2New', 'Technical4New', 'FinancialNew',
     'GrowthNew', 'ShareholderNew', 'TechnicalHighFreq1', 'Risk&Div']
    # mode = test: conduct a test on the local data; mode = save: indicates either updating or extracting the data
    mode = 'save'
    # If mode = save, you can set update = True to update new up-to-date data instead of loading all data
    update = True

    # Test parameters（only when mode = test）
    # Stock pool: 300, 500, 1000, 1800, 2200, 3000, HighBeta800, LowBeta800, HighBeta1000, LowBeta1000, HighBeta1800, LowBeta1800, HighBeta3000, LowBeta3000, BigValue, MedValue, SmallValue, BigGrowth, MedGrowth, SmallGrowth, all
    stk_range = '1800'
    # ret_idx: 'Open5TWAP' or 'ClosePrice', respectively represent the lagged 2-period return rate of the weighted average price in the first 5 minutes after the opening and the lagged 1-period return rate of the closing price
    ret_idx = 'Open5TWAP'
    # Abnormal regime test: 'rise': rapid growth; 'V': rapid drop and recovery; None: do not enable abnormal regime test
    abn_dates_test = None
    # Test for specific month catagories (for timing)：subset of {-3, -2, -1, 1, 2, 3} or None（Note: abn_dates_test and month_status at least 1 set to None）
    month_status = None
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Create output directory
    os.makedirs(factor_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'figs_{stk_range}'), exist_ok=True)

    # Factor field dictionary storage path
    factor_dict_path = r'D:\code\auto_new\Factor_directory.xlsx' # an excel file containing factor group names and factor names
    # Path for storing the dictionary of month status
    status_dir_path = r'D:\code\auto_new\month.xlsx' # an excel file containing month labels

    if mode == 'save':
        # Data storage
        print("Starting data storage...")
        start_time = time.time()
        run_data_storage(start_date, end_date, factor_dir, factor_dict_path, name_list, update)
        end_time = time.time()
        print(f"Data storage completed in {end_time - start_time:.2f} seconds.")

    if mode == 'test':
        # ICIR calculation
        print("\nStarting ICIR calculation...")
        start_time = time.time()
        run_icir_calculation(start_date, end_date, factor_dir, output_dir, name_list, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path)
        end_time = time.time()
        print(f"ICIR calculation completed in {end_time - start_time:.2f} seconds.")

        # Return calculation, graphs and PDF generation
        print("\nStarting return calculation, plotting, and PDF report generation...")
        start_time = time.time()
        run_return_calculation(start_date, end_date, factor_dir, output_dir, name_list, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path)
        end_time = time.time()
        print(f"Return calculation, plotting, and PDF report generation completed in {end_time - start_time:.2f} seconds.")

    print("\nAll processes completed.")

if __name__ == '__main__':
    total_start_time = time.time()
    main()
    total_end_time = time.time()
    print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds.")



