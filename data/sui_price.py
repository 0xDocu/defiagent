import pandas as pd

# Sui Price 정규화
# Sui 일별 데이터는 CoinMarketCap 제공 데이터

INPUT_CSV = 'raw_suiusdc/sui_historical_price.csv'
OUTPUT_CSV = 'normalization/norm_hist_price.csv'

def main():
    
    df = pd.read_csv(INPUT_CSV, sep=';', header=0)
    
    df['date'] = df['timeOpen'].str[:10]
    
    # (O+H+L+C)/4 계산
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    
    # normalization
    _mean = df['avg_price'].mean()
    _std  = df['avg_price'].std()
    df['norm_price'] = (df['avg_price'] - _mean) / _std
    
    out_df = df[['date','avg_price','norm_price']]
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"File saved successfully: {OUTPUT_CSV}")
    print(f"  - mean 평균 = {_mean:.4f}, stdev = {_std:.4f}")
    
if __name__ == "__main__":
    main()
