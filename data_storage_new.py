data_storage_new.py
import pandas as pd
import pymssql
import os
from tqdm import tqdm
import warnings
from datetime import datetime


warnings.filterwarnings('ignore')

def read_sql(query):
    conn = pymssql.connect('192.168.163.10', 'shixi1', 'lhzc0849', 'Quant', tds_version='7.0')
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def get_query_new1010(factors, list_name, start_date, end_date):

    factor = ', a.'.join(factors)
    query = f"""
            SELECT a.TradingDay, a.SecuCode, a.{factor}
            FROM [Factor].[{list_name}] AS a
            WHERE a.TradingDay BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY TradingDay ASC
            """
    return query

def get_stock_index_weight_query(start_date, end_date):

    query = f"""
            SELECT TradingDay, SecuCode, ClosePrice, IndexW300, IndexW500, IndexW1000, ID3000
            FROM [Factor].[StockIndexWeightNew2]
            WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY TradingDay ASC
            """
    return query

def get_beta_pool_query(start_date, end_date):
    # Handling datetime objects or date formats in string format
    if isinstance(start_date, datetime):
        start_date_int = int(start_date.strftime('%Y%m%d'))
    else:
        start_date_int = int(start_date.replace('-', ''))
        
    if isinstance(end_date, datetime):
        end_date_int = int(end_date.strftime('%Y%m%d'))
    else:
        end_date_int = int(end_date.replace('-', ''))

    query = f"""
            SELECT TradingDayInt, CodeInt, HighBeta1800, LowBeta1800, HighBeta800, LowBeta800, HighBeta1000, LowBeta1000, HighBeta3000, LowBeta3000
            FROM [Factor].[StockPoolHLBetaAdj]
            WHERE TradingDayInt BETWEEN {start_date_int} AND {end_date_int}
            ORDER BY TradingDayInt ASC
            """
    return query

def get_bmsgv_pool_query(start_date, end_date):
    # Handling datetime objects or date formats in string format
    if isinstance(start_date, datetime):
        start_date_int = int(start_date.strftime('%Y%m%d'))
    else:
        start_date_int = int(start_date.replace('-', ''))
        
    if isinstance(end_date, datetime):
        end_date_int = int(end_date.strftime('%Y%m%d'))
    else:
        end_date_int = int(end_date.replace('-', ''))

    query = f"""
            SELECT TradingDayInt, CodeInt, BigValue, MedValue, SmallValue, BigGrowth, MedGrowth, SmallGrowth
            FROM [Factor].[StockPoolValueGrowthAdj]
            WHERE TradingDayInt BETWEEN {start_date_int} AND {end_date_int}
            ORDER BY TradingDayInt ASC
            """
    return query

def get_market_info_query(start_date, end_date):

    query = f"""
            SELECT TradingDay, SecuCode, DivR1, DivR2
            FROM [Factor].[MarketInfoNew]
            WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY TradingDay ASC
            """
    return query

def get_risk_factor_query(start_date, end_date):
    # Ensure that the date format is correctly converted
    if isinstance(start_date, str):
        start_date_int = int(start_date.replace('-', ''))
    else:
        start_date_int = int(start_date.strftime('%Y%m%d'))
        
    if isinstance(end_date, str):
        end_date_int = int(end_date.replace('-', ''))
    else:
        end_date_int = int(end_date.strftime('%Y%m%d'))

    query = f"""
            SELECT TradingDayInt, CodeInt AS SecuCode, Beta_orth_ZS, Close_orth_ZS, LM_orth_ZS, SM_orth_ZS, 
                   MVA_orth_ZS, VOL_orth_ZS, DivR2_orth_ZS, EP_orth_ZS, DAR_orth_ZS, ROA_orth_ZS, ORG2_orth_ZS
            FROM [RiskFactor].[LHRM3]
            WHERE TradingDayInt BETWEEN {start_date_int} AND {end_date_int}
            ORDER BY TradingDayInt ASC
            """
    return query

def get_daily_mean_price_query(start_date, end_date):
    # Ensure that the date format is correctly converted
    if isinstance(start_date, str):
        start_date_int = int(start_date.replace('-', ''))
    else:
        start_date_int = int(start_date.strftime('%Y%m%d'))
        
    if isinstance(end_date, str):
        end_date_int = int(end_date.replace('-', ''))
    else:
        end_date_int = int(end_date.strftime('%Y%m%d'))

    query = f"""
            SELECT TradingDayInt, CodeInt AS SecuCode, Open5TWAP
            FROM [Market].[DailyMeanPrice1]
            WHERE TradingDayInt BETWEEN {start_date_int} AND {end_date_int}
            ORDER BY TradingDayInt ASC
            """
    return query

def get_base_data(start_date, end_date, output_dir):
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        year_start = max(current_date, datetime(year, 1, 1))
        year_end = min(end_date, datetime(year, 12, 31))
        
        queries = [
            ('stock_index_weight', get_stock_index_weight_query(year_start, year_end)),
            ('daily_mean_price', get_daily_mean_price_query(year_start, year_end)),
            ('beta_pool', get_beta_pool_query(year_start, year_end)),
            ('bmsgv_pool', get_bmsgv_pool_query(year_start, year_end))
        ]
        for name, query in tqdm(queries, desc=f"Fetching data for base columns {year}", leave=False):           

            data = read_sql(query)

            # Rename first if column name is TradingDayInt
            if 'TradingDayInt' in data.columns:
                data['TradingDay'] = pd.to_datetime(data['TradingDayInt'].astype(str), format='%Y%m%d')
                data = data.drop(columns={'TradingDayInt'})
            else:
                data['TradingDay'] = pd.to_datetime(data['TradingDay'])

            if 'CodeInt' in data.columns:
                data = data.rename(columns={'CodeInt':'SecuCode'})
            data['SecuCode'] = data['SecuCode'].astype(str).str.zfill(6)

            # Convert to float32 for memory optimization
            data = data.apply(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)

            print(f"Successfully fetched {len(data['TradingDay'].unique())} trading days for {name}")

            # Construct a complete directory path
            dir_path = os.path.join(output_dir, str(year))

            # Ensure directory exists
            os.makedirs(dir_path, exist_ok=True)

            # Set path and save
            save_path = os.path.join(dir_path, f'{name}.parquet')
            data.to_parquet(save_path)

        current_date = datetime(year + 1, 1, 1)
    print('Base columns loaded from database')

def fetch_and_save_risk_div_data(year, start_date, end_date, output_dir, update):

    year_dir = os.path.join(output_dir, str(year))
    file_path = os.path.join(year_dir, f'Factors_Risk&Div_banks_all.csv')
    os.makedirs(year_dir, exist_ok=True)    

    data_dict = {}

    if update and year >= 2024:

        queries = [
        ('risk_factor', get_risk_factor_query(start_date, end_date)),
        ('market_info', get_market_info_query(start_date, end_date)),
        ('stock_index_weight', get_stock_index_weight_query(start_date, end_date)),
        ('daily_mean_price', get_daily_mean_price_query(start_date, end_date))
        ]

        if os.path.exists(file_path):
            # Read the last date of the existing file
            existing_dates = pd.read_csv(file_path, usecols=['TradingDay'])
            last_date = pd.to_datetime(existing_dates['TradingDay']).max()
            new_start_date = last_date + pd.Timedelta(days=1)

            if pd.to_datetime(end_date) <= last_date:
                print(f"Data already up to date until {last_date}")
                return
            
            # Only obtain new data
            for name, query in tqdm(queries, desc=f"Updating data for Risk&Div {year}", leave=False):
                modified_query = query.replace(str(start_date), str(new_start_date))
                print(f"\n{name}: Fetching new data from {new_start_date} to {end_date}")
                new_data = read_sql(modified_query)

                # Standardized data format
                if 'TradingDayInt' in new_data.columns:
                    new_data['TradingDay'] = pd.to_datetime(new_data['TradingDayInt'].astype(str), format='%Y%m%d')
                    new_data = new_data.drop(columns=['TradingDayInt'])
                else:
                    new_data['TradingDay'] = pd.to_datetime(new_data['TradingDay'], format='%Y%m%d')

                if 'CodeInt' in new_data.columns:
                    new_data = new_data.rename(columns={'CodeInt': 'SecuCode'})
                new_data['SecuCode'] = new_data['SecuCode'].astype(str).str.zfill(6)

                # Read old data and merge
                if name in ['beta_pool', 'bmsgv_pool']:
                    old_file = os.path.join(year_dir, f'{name}.parquet')
                    if os.path.exists(old_file):
                        old_data = pd.read_parquet(old_file)
                        updated_data = pd.concat([old_data, new_data], ignore_index=True)
                        updated_path = os.path.join(year_dir, f'{name}.csv')
                        updated_data.to_csv(updated_path)
                        del updated_data
                else:
                    data_dict[name] = new_data
            
            stock_index_weight = data_dict['stock_index_weight']
            daily_mean_price = data_dict['daily_mean_price']
            market_info = data_dict['market_info']
            risk_factor = data_dict['risk_factor']

            # Merge Risk and Div factor data
            risk_div_df = market_info.merge(risk_factor, on=['TradingDay', 'SecuCode'], how='inner')
            risk_div_df = risk_div_df.merge(stock_index_weight, on=['TradingDay', 'SecuCode'], how='inner')
            risk_div_df = risk_div_df.merge(daily_mean_price[['TradingDay', 'SecuCode', 'Open5TWAP']], on=['TradingDay', 'SecuCode'], how='inner')
            risk_div_df = risk_div_df.sort_values(['TradingDay', 'SecuCode'])
            del data_dict, stock_index_weight, daily_mean_price, market_info, risk_factor

            # Handle risk_div_df concat
            old_df = pd.read_csv(file_path)
            old_df = old_df.sort_values(['TradingDay', 'SecuCode'])

            common_columns = old_df.columns
            risk_div_df = risk_div_df[common_columns]

            risk_div_df['TradingDay'] = risk_div_df['TradingDay'].dt.date
            risk_div_df = pd.concat([old_df, risk_div_df], axis=0)

            # Save data
            print(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            risk_div_df.to_csv(file_path, index=False)

    else:

        queries = [
        ('risk_factor', get_risk_factor_query(start_date, end_date)),
        ('market_info', get_market_info_query(start_date, end_date)),
        ]

        for name, query in tqdm(queries, desc=f"Fetching data for Risk&Div {year}", leave=False):           

            # If update=False or file does not exist，get all data from database
            data = read_sql(query)
            # Convert to float32 for memory optimization
            data = data.apply(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)

            # Rename first if column name is TradingDayInt
            if 'TradingDayInt' in data.columns:
                data['TradingDay'] = pd.to_datetime(data['TradingDayInt'].astype(str), format='%Y%m%d')
                data = data.drop(columns={'TradingDayInt'})
            else:
                data['TradingDay'] = pd.to_datetime(data['TradingDay'])

            if 'CodeInt' in data.columns:
                data = data.rename(columns={'CodeInt':'SecuCode'})
            data['SecuCode'] = data['SecuCode'].astype(str).str.zfill(6)

            data_dict[name] = data

        stock_index_weight_path = os.path.join(output_dir, str(year), 'stock_index_weight.parquet')
        stock_index_weight = pd.read_parquet(stock_index_weight_path)
        daily_mean_price_path = os.path.join(output_dir, str(year), 'daily_mean_price.parquet')
        daily_mean_price = pd.read_parquet(daily_mean_price_path)
        market_info = data_dict['market_info']
        risk_factor = data_dict['risk_factor']

        # Merge Risk and Div factor data
        risk_div_df = market_info.merge(risk_factor, on=['TradingDay', 'SecuCode'], how='inner')
        risk_div_df = risk_div_df.merge(stock_index_weight, on=['TradingDay', 'SecuCode'], how='inner')
        risk_div_df = risk_div_df.merge(daily_mean_price[['TradingDay', 'SecuCode', 'Open5TWAP']], on=['TradingDay', 'SecuCode'], how='inner')

        # Save data
        risk_div_df = risk_div_df.sort_values(['TradingDay', 'SecuCode'])
        risk_div_df.to_csv(file_path, index=False)

def fetch_and_save_data_update(args):
    list_name, year, start_date, end_date, factor_dict_path, output_dir, update = args

    # Remove 'New'
    list_name_for_filter = list_name[:-3] if list_name.endswith('New') else list_name

    type_df = pd.read_excel(factor_dict_path)
    factors = type_df[(type_df['table'] == list_name_for_filter) & (type_df['data_type'] == 'decimal')]['column'].tolist()

    year_dir = os.path.join(output_dir, str(year))
    file_path = os.path.join(year_dir, f'Factors_{list_name}_banks_all.csv')
    os.makedirs(year_dir, exist_ok=True)

    data_dict = {}

    if update and year >= 2024:

        queries = [
        ('factor_info', get_query_new1010(factors, list_name, start_date, end_date)),
        ('stock_index_weight', get_stock_index_weight_query(start_date, end_date)),
        ('daily_mean_price', get_daily_mean_price_query(start_date, end_date)),
        ]

        if os.path.exists(file_path):
            # Read the last date of the existing file
            existing_dates = pd.read_csv(file_path, usecols=['TradingDay'])
            last_date = pd.to_datetime(existing_dates['TradingDay']).max()
            new_start_date = last_date + pd.Timedelta(days=1)
            # print(new_start_date)

            if pd.to_datetime(end_date) <= last_date:
                print(f"Data already up to date until {last_date}")
                return
            
            # Obtain new data
            for name, query in tqdm(queries, desc=f"Updating data for {list_name} {year}", leave=False):
                modified_query = query.replace(str(start_date), str(new_start_date))
                print(f"\n{name}: Fetching new data from {new_start_date} to {end_date}")
                new_data = read_sql(modified_query)

                # Standardize data format
                if 'TradingDayInt' in new_data.columns:
                    new_data['TradingDay'] = pd.to_datetime(new_data['TradingDayInt'].astype(str), format='%Y%m%d')
                    new_data = new_data.drop(columns=['TradingDayInt'])
                else:
                    new_data['TradingDay'] = pd.to_datetime(new_data['TradingDay'], format='%Y%m%d')

                if 'CodeInt' in new_data.columns:
                    new_data = new_data.rename(columns={'CodeInt': 'SecuCode'})
                new_data['SecuCode'] = new_data['SecuCode'].astype(str).str.zfill(6)

                # Read old data and merge
                if name in ['beta_pool', 'bmsgv_pool']:
                    old_file = os.path.join(year_dir, f'{name}.parquet')
                    if os.path.exists(old_file):
                        old_data = pd.read_parquet(old_file)
                        updated_data = pd.concat([old_data, new_data], ignore_index=True)
                        updated_path = os.path.join(year_dir, f'{name}.csv')
                        updated_data.to_csv(updated_path)
                        del updated_data
                else:
                    data_dict[name] = new_data
            
            factor_info = data_dict['factor_info']
            stock_index_weight = data_dict['stock_index_weight']
            daily_mean_price = data_dict['daily_mean_price']

            # Merge data
            all_df = factor_info.merge(stock_index_weight, on=['TradingDay', 'SecuCode'], how='inner')
            all_df = all_df.merge(daily_mean_price[['TradingDay', 'SecuCode', 'Open5TWAP']], on=['TradingDay', 'SecuCode'], how='inner')
            all_df = all_df.sort_values(['TradingDay', 'SecuCode'])
            del data_dict, factor_info, stock_index_weight, daily_mean_price

            # Handle all_df concat
            old_df = pd.read_csv(file_path)
            old_df = old_df.sort_values(['TradingDay', 'SecuCode'])

            common_columns = old_df.columns
            all_df = all_df[common_columns]
            all_df['TradingDay'] = all_df['TradingDay'].dt.date
            all_df = pd.concat([old_df, all_df], axis=0)

            # Save data
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            all_df.to_csv(file_path, index=False)

    else:

        queries = [
        ('factor_info', get_query_new1010(factors, list_name, start_date, end_date)),
        ]

        for name, query in tqdm(queries, desc=f"Fetching data for {list_name} {year}", leave=False):           

            # If update=False or file does not exist，get all data from database
            data = read_sql(query)
            # Convert to float32 for memory optimization
            data = data.apply(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)

            # Rename first if column name is TradingDayInt
            if 'TradingDayInt' in data.columns:
                data['TradingDay'] = pd.to_datetime(data['TradingDayInt'].astype(str), format='%Y%m%d')
                data = data.drop(columns={'TradingDayInt'})
            else:
                data['TradingDay'] = pd.to_datetime(data['TradingDay'])

            if 'CodeInt' in data.columns:
                data = data.rename(columns={'CodeInt':'SecuCode'})
            data['SecuCode'] = data['SecuCode'].astype(str).str.zfill(6)

            data_dict[name] = data

        stock_index_weight_path = os.path.join(output_dir, str(year), 'stock_index_weight.parquet')
        stock_index_weight = pd.read_parquet(stock_index_weight_path)
        daily_mean_price_path = os.path.join(output_dir, str(year), 'daily_mean_price.parquet')
        daily_mean_price = pd.read_parquet(daily_mean_price_path)
        factor_info = data_dict['factor_info']

        # Merge data
        all_df = factor_info.merge(stock_index_weight, on=['TradingDay', 'SecuCode'], how='inner')
        all_df = all_df.merge(daily_mean_price[['TradingDay', 'SecuCode', 'Open5TWAP']], on=['TradingDay', 'SecuCode'], how='inner')
        del stock_index_weight, daily_mean_price, factor_info

        # Save data
        all_df = all_df.sort_values(['TradingDay', 'SecuCode'])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        all_df.to_csv(file_path, index=False)

def run_data_storage(start_date, end_date, output_dir, factor_dict_path, name_list=None, update=False, batch_size=5):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")  

    if update == True:
        if not name_list:
            # If name_list is empty，only process Risk&Div data
            fetch_and_save_risk_div_data(
                2024,
                '2024-01-01',
                end_date.strftime("%Y-%m-%d"),
                output_dir,
                update
            )
        else:
            # Batching
            def chunks(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
            
            name_batches = list(chunks(name_list, batch_size))
            
            for batch_idx, name_batch in enumerate(name_batches, 1):
                print(f"\nProcessing batch {batch_idx}/{len(name_batches)}")
                for name in tqdm(name_batch, desc=f"Batch {batch_idx}/{len(name_batches)}"):
                    args_list = [name, 2024, '2024-01-01', 
                        end_date.strftime("%Y-%m-%d"), 
                        factor_dict_path, output_dir, update]
                    fetch_and_save_data_update(args_list                          
                    )
            
            fetch_and_save_risk_div_data(
                2024,
                '2024-01-01',
                end_date.strftime("%Y-%m-%d"),
                output_dir,
                update
            )
            
    elif update == False:

        get_base_data(start_date, end_date, output_dir)

        if not name_list:
            # If name_list is empty，only process Risk&Div data
            current_date = start_date
            while current_date <= end_date:
                year = current_date.year
                year_start = max(current_date, datetime(year, 1, 1))
                year_end = min(end_date, datetime(year, 12, 31))
                
                fetch_and_save_risk_div_data(
                    year,
                    year_start.strftime("%Y-%m-%d"),
                    year_end.strftime("%Y-%m-%d"),
                    output_dir,
                    update
                )
                
                current_date = datetime(year + 1, 1, 1)
        else:
            def chunks(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
            
            name_batches = list(chunks(name_list, batch_size))
            
            for batch_idx, name_batch in enumerate(name_batches, 1):
                print(f"\nProcessing batch {batch_idx}/{len(name_batches)}")
                
                current_date = start_date
                while current_date <= end_date:
                    year = current_date.year
                    year_start = max(current_date, datetime(year, 1, 1))
                    year_end = min(end_date, datetime(year, 12, 31))
                    
                    for name in tqdm(name_batch, desc=f"Batch {batch_idx}/{len(name_batches)} - Year {year}"):
                        args_list = [name, year, 
                            year_start.strftime("%Y-%m-%d"), 
                            year_end.strftime("%Y-%m-%d"), 
                            factor_dict_path, output_dir, update]
                        fetch_and_save_data_update(args_list
                        )
                    
                    current_date = datetime(year + 1, 1, 1)
                    
            # current_date = start_date
            # while current_date <= end_date:
            #     year = current_date.year
            #     year_start = max(current_date, datetime(year, 1, 1))
            #     year_end = min(end_date, datetime(year, 12, 31))
                
            #     fetch_and_save_risk_div_data(
            #         year,
            #         year_start.strftime("%Y-%m-%d"),
            #         year_end.strftime("%Y-%m-%d"),
            #         output_dir,
            #         update
            #     )
                
            #     current_date = datetime(year + 1, 1, 1)
    else:
        print('Update variable not valid')
    print("\nData storage and processing completed.")









