import yfinance as yf


start_date = '2006-01-01'
end_date = '2018-01-01'
df = yf.download('329180.KS')


print(df)