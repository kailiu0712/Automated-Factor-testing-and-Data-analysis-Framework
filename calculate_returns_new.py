# calculate_returns_new.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import re
from pandas import Interval
from PyPDF2 import PdfMerger


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

def process_factor(args):
    data, factor, abn_dates_test = args
    # Columns that are not involved in the calculation
    drop_list = ['Unnamed: 0', 'TradingDayInt', 'ReportDateInt', 'IndexW300', 'IndexW500', 'IndexW1000', 'ID3000', 'HighBeta1800',
                  'LowBeta1800', 'BigGrowth', 'MedGrowth', 'SmallGrowth', 'BigValue', 'MedValue', 'SmallValue',
                  'UP_DOWN_LIMIT_STATUS', 'LOWEST_HIGHEST_STATUS', ' AptOutFlowRatio']
    if factor in drop_list:
        return factor, None
    df = data.copy()
    df2 = df.dropna(subset=[factor, 'next_return'])
    
    factor_values = df2.groupby('TradingDay')[factor].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
    
    grouped = df2.groupby(['TradingDay', factor_values])
    daily_means = df2.groupby('TradingDay')['next_return'].mean()
    
    # Calculate the returns for each group
    group_returns = grouped['next_return'].mean().unstack()
    
    if abn_dates_test == 'rise':
        # Subtract the average return for that day from each row (each trading day)
        for day in group_returns.index:
            group_returns.loc[day] = group_returns.loc[day] - daily_means[day]
    
    total_return_mean = group_returns.mean()
    
    return factor, total_return_mean

def cal_return(data, factors, abn_dates_test, batch_size=10):
    total_return_mean = {}
    
    # Batch processing factor
    for i in range(0, len(factors), batch_size):
        batch_factors = factors[i:i+batch_size]
        
        for factor in batch_factors:
            try:
                result = process_factor((data, factor, abn_dates_test))
                
                factor_name, total = result
                
                if total is not None:
                    total_return_mean[factor_name] = total
                    
            except Exception as e:
                print(f"Error processing factor {factor}: {str(e)}")
                continue
                
    return total_return_mean

def plot(data, factor, output_dir):
    if data is None:
        return
    data.index = data.index.astype(int)
    plt.figure(figsize=(6, 4))
    plt.bar(data.index, data.values)
    plt.title(f'{factor} Factor Returns')
    plt.xlabel('Quantile')
    plt.ylabel('Return')
    plt.savefig(os.path.join(output_dir, f'{factor}_returns.png'))
    plt.close()

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# Write PDF
def create_pdf_report(factor_data, output_dir, name, start_date, end_date, stk_range, abn_dates_test, month_status):
    if month_status is not None:
        status_str = ','.join([str(x) for x in month_status])
        pdf_path = os.path.join(output_dir, f'{name}_factor_report_{status_str}_{stk_range}.pdf')
    if abn_dates_test == 'rise' or abn_dates_test == 'V':
        pdf_path = os.path.join(output_dir, f'{name}_factor_report_{abn_dates_test}_{stk_range}.pdf')
    if abn_dates_test == None and month_status == None:
        pdf_path = os.path.join(output_dir, f'{name}_factor_report_{start_date}-{end_date}_{stk_range}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    if month_status is not None:
        story.append(Paragraph(f"{name} Factor Report {status_str} {stk_range}", styles['Title']))
    if abn_dates_test == 'rise' or abn_dates_test == 'V':
        story.append(Paragraph(f"{name} Factor Report {abn_dates_test} {stk_range}", styles['Title']))
    if abn_dates_test == None and month_status == None:
        story.append(Paragraph(f"{name} Factor Report ({start_date} - {end_date}) {stk_range}", styles['Title']))
    story.append(Spacer(1, 12))

    factors = sorted(list(factor_data.keys()), key=natural_sort_key)
    for i in range(0, len(factors), 6):
        table_data = []
        for j in range(i, min(i+6, len(factors)), 2):
            row = []
            for k in range(2):
                if j+k < len(factors):
                    factor = factors[j+k]
                    total_return, icir = factor_data[factor]
                    img_path = os.path.join(output_dir, f'figs_{stk_range}', f'{factor}_returns.png')
                    if os.path.exists(img_path):
                        img = Image(img_path, width=3*inch, height=2*inch)
                        row.append([Paragraph(f"{factor} (ICIR: {icir:.4f})", styles['Normal']), img])
                    else:
                        row.append([Paragraph(f"Image not found for {factor}", styles['Normal']), ''])
            table_data.append(row)
        
        table = Table(table_data, colWidths=[3*inch]*2)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(table)
        story.append(PageBreak())

    doc.build(story)

# Get base columns
def get_base_columns():
    return ['TradingDay', 'SecuCode', 'ClosePrice', 'Open5TWAP', 'IndexW300', 
            'IndexW500', 'IndexW1000', 'ID3000']

def read_factor_batch(file_path, base_columns, factor_batch):
    cols_to_read = base_columns + factor_batch
    df = pd.read_csv(file_path, usecols=cols_to_read)
    df = df.apply(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)
    return df

def gen_plot_pdf(name, all_pdfs, start_date, end_date, input_dir, output_dir, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path):
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    if month_status is not None:
        status_str = ','.join([str(x) for x in month_status])
        icir_file = os.path.join(output_dir, f"RankICIR_{status_str}_{stk_range}_{name}.csv")
    if abn_dates_test == 'rise' or abn_dates_test == 'V':
        icir_file = os.path.join(output_dir, f"RankICIR_{abn_dates_test}_{stk_range}_{name}.csv")
    if abn_dates_test == None and month_status == None:
        icir_file = os.path.join(output_dir, f"RankICIR{start_date}-{end_date}_{stk_range}_{name}.csv")
    
    if not os.path.exists(icir_file):
        print(f"Error: ICIR file not found for {name}")
        return

    data = pd.read_csv(icir_file, encoding='gbk')
    all_factors = data.iloc[:, 0].values
    icir_dict = dict(zip(data['Column'], data['ICIR']))
    base_columns = get_base_columns()

    # Divide the factors into smaller batches
    batch_size = 10
    factor_batches = [list(all_factors[i:i + batch_size]) 
                     for i in range(0, len(all_factors), batch_size)]
    print(f"\nSplit {name} factors into {len(factor_batches)} batches (size: {batch_size})")

    factor_data = {}
    
    # Show process
    for batch_idx, factor_batch in tqdm(enumerate(factor_batches), 
                                      total=len(factor_batches),
                                      desc="Processing factor batches"):
        print(f"\nProcessing batch {batch_idx + 1}/{len(factor_batches)}")
        
        df_list = []
        for year in range(start_year, end_year + 1):
            year_dir = os.path.join(input_dir, str(year))
            factor_file = os.path.join(year_dir, f'Factors_{name}_banks_all.csv')
            
            df = read_factor_batch(factor_file, base_columns, factor_batch)
            df_list.append(df)

        if not df_list:
            print(f"Warning: No data found for {name}")
            continue

        df = pd.concat(df_list, ignore_index=True)
        df['TradingDay'] = pd.to_datetime(df['TradingDay'])

        # Different return formulas
        if ret_idx == 'Open5TWAP':
            df['PctChange'] = df.groupby('SecuCode')['Open5TWAP'].pct_change()
            df['next_return'] = df.groupby('SecuCode')['PctChange'].shift(-2)
        elif ret_idx == 'ClosePrice':
            df['PctChange'] = df.groupby('SecuCode')['ClosePrice'].pct_change()
            df['next_return'] = df.groupby('SecuCode')['PctChange'].shift(-1)

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

        total_mean = cal_return(df, factor_batch, abn_dates_test)
        
        for factor in factor_batch:
            if factor in total_mean and total_mean[factor] is not None:
                plot(total_mean[factor], factor, os.path.join(output_dir, f'figs_{stk_range}'))
                factor_data[factor] = (total_mean[factor], icir_dict.get(factor, 0))

        # Clear cache
        del df_list, df
        
    print(f"\nGenerating {name} PDF report...")
    create_pdf_report(factor_data, output_dir, name, start_date, end_date, stk_range, abn_dates_test, month_status)
    if month_status is not None:
        status_str = ','.join([str(x) for x in month_status])
        all_pdfs.append(os.path.join(output_dir, f'{name}_factor_report_{status_str}_{stk_range}.pdf'))
    if abn_dates_test == 'rise' or abn_dates_test == 'V':
        all_pdfs.append(os.path.join(output_dir, f'{name}_factor_report_{abn_dates_test}_{stk_range}.pdf'))
    if abn_dates_test == None and month_status == None:
        all_pdfs.append(os.path.join(output_dir, f'{name}_factor_report_{start_date}-{end_date}_{stk_range}.pdf'))
    return all_pdfs

def run_return_calculation(start_date, end_date, input_dir, output_dir, name_list, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path):
    all_pdfs = []

    for name in name_list:
        all_pdfs = gen_plot_pdf(name, all_pdfs, start_date, end_date, input_dir, output_dir, stk_range, ret_idx, abn_dates_test, month_status, status_dir_path)

    # Create PdfMerger object
    merger = PdfMerger()
    for pdf_path in all_pdfs:
        merger.append(pdf_path)

    # Construct the complete path for the output PDF
    pdf_name_parts = [f'Factor_test_{start_date}-{end_date}']

    if abn_dates_test is not None:
        pdf_name_parts.append(str(abn_dates_test))

    if month_status is not None:
        status_str = ','.join([str(x) for x in month_status])
        pdf_name_parts.append(status_str)

    pdf_name_parts.append(stk_range)
    output_pdf_path = os.path.join(output_dir, '_'.join(pdf_name_parts) + '.pdf')

    # Save the merged file as a new PDF
    merger.write(output_pdf_path)

    # Close merger
    merger.close()




