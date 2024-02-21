import FinanceDataReader as fdr
import pandas as pd

symbol_list = ['KRX']

for symbol in symbol_list:
    df = fdr.StockListing(symbol)
    
    df.to_csv(f'{symbol}_stock_list.csv', index=False, encoding='utf-8-sig')