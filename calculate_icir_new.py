# calculate_icir_new.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from numba import jit
import gc
from pandas import Interval

# Abnormal regime test
def filter_extreme_periods_rise(df):
    
    intervals = [
    Interval(pd.Timestamp('2010-09-30'), pd.Timestamp('2010-10-15'), closed='both'),
    Interval(pd.Timestamp('2012-12-04'), pd.Timestamp('2012-12-14'), closed='both'),
    Interval(pd.Timestamp('2015-03-10'), pd.Timestamp('2015-06-12'), closed='both'),
    Interval(pd.Timestamp('2018-02-12'), pd.Timestamp('2018-02-26'), closed='both'),
    Interval(pd.Timestamp('2019-02-01'), pd.Timestamp('2019-02-25'), closed='both'),
    Interval(pd.Timestamp('2020-02-04'), pd.Timestamp('2020-02-20'), closed='both'),
    Interval(pd.Timestamp('2020-06-30'), pd.Timestamp('2020-07-09'), closed='both'),
    Interval(pd.Timestamp('2021-02-08'), pd.Timestamp('2021-02-10'), closed='both'),
    Interval(pd.Timestamp('2021-07-29'), pd.Timestamp('2021-08-04'), closed='both'),
    Interval(pd.Timestamp('2024-09-24'), pd.Timestamp('2024-10-08'), closed='both'),
    ]
    
    mask = False
    for interval in intervals:
        mask = mask | ((df['TradingDay'] >= interval.left) & (df['TradingDay'] <= interval.right))
    
    return df[mask]

def filter_extreme_periods_V(df):
    
    intervals = [
    Interval(pd.Timestamp('2011-12-02'), pd.Timestamp('2012-01-10'), closed='both'),
    Interval(pd.Timestamp('2012-03-28'), pd.Timestamp('2012-04-20'), closed='both'),
    Interval(pd.Timestamp('2015-07-01'), pd.Timestamp('2015-07-23'), closed='both'),
    Interval(pd.Timestamp('2016-01-20'), pd.Timestamp('2016-02-22'), closed='both'),
    Interval(pd.Timestamp('2018-02-06'), pd.Timestamp('2018-02-26'), closed='both'),
    Interval(pd.Timestamp('2020-01-21'), pd.Timestamp('2020-02-20'), closed='both'),
    Interval(pd.Timestamp('2021-07-23'), pd.Timestamp('2021-08-04'), closed='both'),
    Interval(pd.Timestamp('2022-03-02'), pd.Timestamp('2022-03-18'), closed='both'),
    Interval(pd.Timestamp('2022-04-20'), pd.Timestamp('2022-04-29'), closed='both'),
    Interval(pd.Timestamp('2023-10-13'), pd.Timestamp('2023-10-30'), closed='both'),
    Interval(pd.Timestamp('2024-01-26'), pd.Timestamp('2024-02-08'), closed='both'),
    Interval(pd.Timestamp('2024-10-09'), pd.Timestamp('2024-11-07'), closed='both')
    ]
    
    mask = False
    for interval in intervals:
        mask = mask | ((df['TradingDay'] >= interval.left) & (df['TradingDay'] <= interval.right))
    
    return df[mask]

def filter_by_month_status(df, month_status, dir_path):
    if len(month_status) == 0:
        return df 
    # Read the status dictionary
    status_df = pd.read_excel(dir_path)
    
    selected_months = status_df[status_df['state'].isin(month_status)]['month'].astype(str).tolist()
    df['month'] = df['TradingDay'].dt.strftime('%Y%m')
    filtered_df = df[df['month'].isin(selected_months)].copy()
    filtered_df.drop('month', axis=1, inplace=True)
    
    return filtered_df

# Screening of constituent stocks and stock pools
def select_stock(stk_range, df, start_year, end_year, input_dir):
    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df['SecuCode'] = df['SecuCode'].astype(str).str.zfill(6)

    basic_pools = {
        '1800': lambda df: df[(df['IndexW300'] > 0) | (df['IndexW500'] > 0) | (df['IndexW1000'] > 0)],
        '300': lambda df: df[df['IndexW300'] > 0],
        '500': lambda df: df[df['IndexW500'] > 0],
        '800': lambda df: df[(df['IndexW300'] > 0) | (df['IndexW500'] > 0)],
        '3000': lambda df: df[df['ID3000'] > 0],
        '2200': lambda df: df[(df['IndexW300'] == 0) & (df['IndexW500'] == 0) & (df['ID3000'] > 0)],
        'all': lambda df: df
    }
    
    if stk_range in basic_pools:
        return basic_pools[stk_range](df)
    
    else:
        if stk_range in ['HighBeta800', 'LowBeta800', 'HighBeta1000', 'LowBeta1000','HighBeta1800', 'LowBeta1800','HighBeta3000', 'LowBeta3000']:
            pool_df_list = []
            for year in range(start_year, end_year + 1):
                year_dir = os.path.join(input_dir, str(year))
                factor_file = os.path.join(year_dir, 'beta_pool.parquet')
                df1 = pd.read_parquet(factor_file, columns=['TradingDay', 'SecuCode', stk_range])
                pool_df_list.append(df1)
            
            if not pool_df_list:
                print(f"No data found for beta pool")
                return pd.DataFrame()
            
            pool_df = pd.concat(pool_df_list, ignore_index=True)
            pool_df['TradingDay'] = pd.to_datetime(pool_df['TradingDay'])
            pool_df['SecuCode'] = pool_df['SecuCode'].astype(str).str.zfill(6)

            beta_mask = pool_df[pool_df[stk_range] > 0][['TradingDay', 'SecuCode']]
            df = df.merge(beta_mask, on=['TradingDay', 'SecuCode'], how='inner')
            del beta_mask, pool_df, pool_df_list, df1
            return df
        
        elif stk_range in ['BigGrowth', 'MedGrowth', 'SmallGrowth', 'BigValue', 'MedValue', 'SmallValue']:
            pool_df_list = []
            for year in range(start_year, end_year + 1):
                year_dir = os.path.join(input_dir, str(year))
                factor_file = os.path.join(year_dir, 'bmsgv_pool.parquet')
                df1 = pd.read_parquet(factor_file, columns=['TradingDay', 'SecuCode', stk_range])
                pool_df_list.append(df1)
            
            if not pool_df_list:
                print(f"No data found for bmsgv pool")
                return pd.DataFrame()
            
            pool_df = pd.concat(pool_df_list, ignore_index=True)
            pool_df['TradingDay'] = pd.to_datetime(pool_df['TradingDay'])
            pool_df['SecuCode'] = pool_df['SecuCode'].astype(str).str.zfill(6)

            bmsgv_mask = pool_df[pool_df[stk_range] > 0][['TradingDay', 'SecuCode']]
            df = df.merge(bmsgv_mask, on=['TradingDay', 'SecuCode'], how='inner')
            del bmsgv_mask, pool_df, pool_df_list, df1
            return df
        
        else:
            print('stock range not valid')
            return pd.DataFrame()


def run_icir_calculation(start_date, end_date, input_dir, output_dir, name_list, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path):
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    @jit(nopython=True)
    def fast_spearman(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    def calc_icir(df, col):
        grouped = df.groupby('TradingDay')
        ic = grouped.apply(lambda g: fast_spearman(g[col].rank(method='first').values, g['next_return_rank'].values))
        ic = ic.dropna()
        if len(ic) == 0:
            return np.nan, np.nan, np.nan
        return ic.mean(), ic.std(), ic.mean() / ic.std()

    def process_columns(df, columns):
        results = {}
        for col in columns:
            ic_mean, ic_std, icir = calc_icir(df, col)
            results[col] = {'IC_Mean': ic_mean, 'IC_Std': ic_std, 'ICIR': icir}
        return results

    def process_factor(name, ret_idx, abn_dates_test, month_status, status_dir_path):
        print(f"\nStart processing factor type {name} ...")
        
        # Base columns, read every time
        base_cols = ['TradingDay', 'SecuCode', 'ClosePrice', 'Open5TWAP', 'IndexW300', 'IndexW500', 'IndexW1000', 'ID3000']
        
        def get_all_columns(file_path):
            return pd.read_csv(file_path, nrows=0).columns.tolist()
        
        def process_batch(df, factor_cols, abn_dates_test, month_status, status_dir_path):
            df['TradingDay'] = pd.to_datetime(df['TradingDay'])
            df['year'] = df['TradingDay'].dt.year
            
            # Different return formulas
            if ret_idx == 'Open5TWAP':
                df['PctChange'] = df.groupby('SecuCode')['Open5TWAP'].pct_change()
                df['next_return'] = df.groupby('SecuCode')['PctChange'].shift(-2)
            elif ret_idx == 'ClosePrice':
                df['PctChange'] = df.groupby('SecuCode')['ClosePrice'].pct_change()
                df['next_return'] = df.groupby('SecuCode')['PctChange'].shift(-1)
                
            df.dropna(subset=['next_return'], inplace=True)
            # Abnormal regime test
            if abn_dates_test == 'rise':
                df = filter_extreme_periods_rise(df)
            elif abn_dates_test == 'V':
                df = filter_extreme_periods_V(df)
            elif abn_dates_test == None:
                pass
            else:
                print('abn_dates_test not valid')
            # Month status test
            if month_status is not None:
                df = filter_by_month_status(df, month_status, status_dir_path)
            else:
                pass
            # Filter date period
            df = df[(df['TradingDay'] >= start_date) & (df['TradingDay'] <= end_date)]
            # Filter constituent stocks
            df = select_stock(stk_range, df, start_year, end_year, input_dir)
            df['next_return_rank'] = df.groupby('TradingDay')['next_return'].rank()
            
            results = process_columns(df, factor_cols)
            return results

        # Obtain the files from the first year to determine all the column names
        first_file = os.path.join(input_dir, str(start_year), 'Factors_{}_banks_all.csv'.format(name))
        all_columns = get_all_columns(first_file)
        factor_columns = [col for col in all_columns if col not in base_cols]
        
        # Set batch size
        batch_size = 10
        total_batches = (len(factor_columns) + batch_size - 1) // batch_size
        
        all_results = {}
        
        for batch_idx in tqdm(range(total_batches), desc=f"Processing factor batches"):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(factor_columns))
            current_factor_cols = factor_columns[batch_start:batch_end]
            print(f"\nProcessing {name} batch {batch_idx + 1}/{total_batches} , factor num: {len(current_factor_cols)}")
            
            # When reading the data, only the required columns are read
            cols_to_read = base_cols + current_factor_cols
            
            df_list = []
            for year in range(start_year, end_year + 1):
                year_dir = os.path.join(input_dir, str(year))
                factor_file = os.path.join(year_dir, 'Factors_{}_banks_all.csv'.format(name))
                df = pd.read_csv(factor_file, usecols=cols_to_read)
                # Convert to float32 for memory optimization
                df = df.apply(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)
                df_list.append(df)
            
            if not df_list:
                print(f"Warnings: {name} not found")
                continue
            
            df = pd.concat(df_list, ignore_index=True)
            
            batch_results = process_batch(df, current_factor_cols, abn_dates_test, month_status, status_dir_path)
            all_results.update(batch_results)
            
            # Clear cache
            del df_list, df
            gc.collect()
            
        # Merge all the results and save them
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Column'
        results_df = results_df.round(4)
        if month_status is not None:
            status_str = ','.join([str(x) for x in month_status])
            output_file = os.path.join(output_dir, f'RankICIR_{status_str}_{stk_range}_{name}.csv')
        if abn_dates_test == 'rise' or abn_dates_test == 'V':
            output_file = os.path.join(output_dir, f'RankICIR_{abn_dates_test}_{stk_range}_{name}.csv')
        if abn_dates_test == None and month_status == None:
            output_file = os.path.join(output_dir, f'RankICIR{start_date}-{end_date}_{stk_range}_{name}.csv')
        results_df.to_csv(output_file, encoding='gbk', index=True)
        print(f"ICIR calculation completed! Saved to {output_file}")
        print(f"Result contains {len(results_df)} factor information")

    for name in name_list:
        print(f"Processing {name}")
        process_factor(name, ret_idx, abn_dates_test, month_status, status_dir_path)

    print("ICIR calculation completed for all lists.")



