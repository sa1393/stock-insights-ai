import FinanceDataReader as fdr
import pandas as pd

print('ttt')
symbol_list = ['KRX', 'KOSPI', 'KOSDAQ', 'KONEX']

for symbol in symbol_list:
    df = fdr.StockListing(symbol)
    df.to_csv(f'{symbol}_stock_list.csv', index=False)