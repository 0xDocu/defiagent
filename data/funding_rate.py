import pandas as pd

# Binance SUIUSDC perpetuals funding rates 데이터 정규화
# Binance 제공 데이터
# https://www.binance.com/en/futures/funding-history/perpetual/funding-fee-history

# SUIUSDC의 과거 데이터가 부족하여 SUIUSDT 데이터 사용

INPUT_CSV = 'raw_suiusdc/Binance_Funding Rate _SUIUSDT Perpetual.csv'
# 일자별 평균 + 정규화 결과물
OUTPUT_CSV = 'normalization/norm_funding_rate.csv'

def main():
    df = pd.read_csv(INPUT_CSV)
    
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
    df['Funding Rate'] = df['Funding Rate'].str.replace('%','', regex=False)
    df['Funding Rate'] = pd.to_numeric(df['Funding Rate'], errors='coerce')
    
    # 8시간 단위 데이터를 일별 평균으로 전환
    df['date'] = df['Time'].dt.strftime('%Y-%m-%d')
    daily_df = df.groupby('date', as_index=False)['Funding Rate'].mean()
    
    # 일자별 평균 데이터 저장
    daily_df.to_csv(OUTPUT_CSV, index=False)
    
    # normailzation(z-score)
    _mean = daily_df['Funding Rate'].mean()
    _std  = daily_df['Funding Rate'].std()
    daily_df['funding_scaled'] = (daily_df['Funding Rate'] - _mean) / _std
    
    daily_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"File saved successfully: {OUTPUT_CSV}")
    print(f"  - daily_funding mean={_mean:.6f}, std={_std:.6f}")

if __name__ == '__main__':
    main()
