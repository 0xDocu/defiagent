import json
import pandas as pd
from datetime import datetime

# 학습 대상인 CETUS의 historical APY data가 부족
# 부족한 기간은 다른 DEX인 turbos를 이용

# 파일 경로
CETUS_JSON = 'raw_data/pool_cetus_USDC_SUI.json'
TURBOS_JSON = 'raw_data/pool_turbos_SUIUSDT.json'
OUTPUT_CSV = 'raw_data/merged_cetus_turbos.csv'

def main():
    with open(CETUS_JSON, 'r') as f:
        cetus_raw = json.load(f)
    with open(TURBOS_JSON, 'r') as f:
        turbos_raw = json.load(f)
    
    cetus_data = cetus_raw.get('data', [])
    turbos_data = turbos_raw.get('data', [])
    
    cetus_df = pd.DataFrame(cetus_data)
    turbos_df = pd.DataFrame(turbos_data)
    
    # timestamp -> date 변환
    cetus_df['date'] = pd.to_datetime(cetus_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')
    turbos_df['date'] = pd.to_datetime(turbos_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # 양쪽 정보 취합 준비
    cetus_df = cetus_df[['date','tvlUsd','apy']].rename(columns={
        'tvlUsd':'tvl_cetus',
        'apy':'apy_cetus'
    })
    turbos_df = turbos_df[['date','tvlUsd','apy']].rename(columns={
        'tvlUsd':'tvl_turbos',
        'apy':'apy_turbos'
    })
    
    # date 기준 outer join
    merged = pd.merge(cetus_df, turbos_df, on='date', how='outer')
    
    merged['merged_apy'] = merged.apply(
        lambda row: row['apy_cetus'] if pd.notnull(row['apy_cetus']) else row['apy_turbos'], axis=1
    )
    
    merged['merged_tvl'] = merged.apply(
        lambda row: row['tvl_cetus'] if pd.notnull(row['tvl_cetus']) else row['tvl_turbos'], axis=1
    )
    
    merged.sort_values('date', inplace=True)
    
    merged.to_csv(OUTPUT_CSV, index=False)
    
    print(f"File saved successfully: {OUTPUT_CSV}")
    
if __name__ == '__main__':
    main()
