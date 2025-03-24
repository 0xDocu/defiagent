import pandas as pd

INPUT_CSV = 'raw_data/merged_cetus_turbos.csv'
OUTPUT_CSV = 'normalization/norm_apy_tvl.csv'

def main():
    
    df = pd.read_csv(INPUT_CSV)
    df_subset = df[['date','merged_apy','merged_tvl']].copy()

    df_subset['date'] = df_subset['date'].dt.strftime('%Y-%m-%d')
    
    
    # 3) z-score 정규화
    _mean_apy = df_subset['merged_apy'].mean()
    _std_apy  = df_subset['merged_apy'].std()
    _mean_tvl = df_subset['merged_tvl'].mean()
    _std_tvl  = df_subset['merged_tvl'].std()
    
    df_subset['merged_apy_scaled'] = (df_subset['merged_apy'] - _mean_apy) / _std_apy
    df_subset['merged_tvl_scaled'] = (df_subset['merged_tvl'] - _mean_tvl) / _std_tvl
    
    df_subset.to_csv(OUTPUT_CSV, index=False)
    
    print(f"File saved successfully: {OUTPUT_CSV}")
    print(f"  - merged_apy mean={_mean_apy:.4f}, std={_std_apy:.4f}")
    print(f"  - merged_tvl mean={_mean_tvl:.4f}, std={_std_tvl:.4f}")

if __name__ == '__main__':
    main()
