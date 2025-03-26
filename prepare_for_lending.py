# prepare_for_lending.py
# 목적: dlinear_3.py와 동일한 로직으로 학습 데이터를 읽고,
#       마지막 30일 시계열 가격 -> offchain_preprocess -> (trend, resid) -> 합쳐 60차원 -> 정규화
#       => (sign + magnitude) 형태로 변환 후, input_1.json 저장

import pandas as pd
import numpy as np
import json
import os

DATA_CSV = 'data/dataset/data_1st.csv'
WINDOW_SIZE = 30
HORIZON = 1

SCALE = 10**2  # 소수점 5자리 (예: 1.23456 → magnitude=123456, sign=0)

def offchain_preprocess(prices_1d):
    """
    prices_1d: shape=(window_size,)  ex) length=30
    returns: (trend_seq, resid_seq), each shape=(30,)
    """
    window_size = len(prices_1d)
    trend_seq = np.zeros(window_size)
    for i in range(window_size):
        left = max(0, i-2)
        right= min(window_size, i+3)
        seg = prices_1d[left:right]
        trend_seq[i] = np.mean(seg)
    resid_seq = prices_1d - trend_seq
    return trend_seq, resid_seq

def create_window_multistep(df, price_col, apy_col, window_size, horizon):
    """
    동일한 dlinear_3.py 로직
    """
    values_price = df[price_col].values
    values_apy   = df[apy_col].values
    dates        = df['date'].values
    
    X_list, Y_list, date_label = [], [], []
    N = len(df)
    
    for i in range(N - window_size - horizon + 1):
        x_window = values_price[i : i+window_size]
        y_window = values_apy[i+window_size : i+window_size+horizon]
        label_date = dates[i+window_size]
        
        X_list.append(x_window)
        Y_list.append(y_window)
        date_label.append(label_date)
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    date_label = np.array(date_label)
    
    return X, Y, date_label

def to_fixed_point(values, scale=SCALE):
    """
    values: list or array of floats
    scale : int (10^5 => 소수점 5자리)
    returns: (arr_mag, arr_sign)
    """
    arr_mag = []
    arr_sign = []
    for val in values:
        if val >= 0:
            arr_sign.append(0)
            mag = val * scale
        else:
            arr_sign.append(1)
            mag = -val * scale
        # 반올림 (또는 int()로 버림)
        mag_int = int(round(mag))
        arr_mag.append(mag_int)
    return arr_mag, arr_sign

def main():
    df = pd.read_csv(DATA_CSV)
    df.sort_values('date', inplace=True, ignore_index=True)
    print("df.shape =", df.shape)

    # dlinear_3.py: X_raw, Y => create_window_multistep
    X_raw, Y, date_label = create_window_multistep(
        df, 'price', 'apy', WINDOW_SIZE, HORIZON
    )
    print("Initial X_raw shape =", X_raw.shape)
    print("Y shape =", Y.shape)

    # offchain_preprocess -> merged(60,)
    # X2
    X2_list = []
    for x_window in X_raw:
        trend_seq, resid_seq = offchain_preprocess(x_window)
        merged_2d = np.concatenate([trend_seq, resid_seq], axis=0)  # shape=(60,)
        X2_list.append(merged_2d)
    X2 = np.array(X2_list)
    print("X2 shape =", X2.shape)

    # train/val/test split
    N = len(X2)
    train_end = int(N*0.7)
    # val_end   = int(N*0.85)  # 여기서는 굳이 안씀
    X2_train = X2[:train_end]

    print("Train size:", X2_train.shape)

    # mean, std from train set
    _mean = X2_train.mean(axis=0)
    _std  = X2_train.std(axis=0) + 1e-8  # small offset to avoid 0-div

    def normalize(x, m, s):
        return (x - m) / s

    # ----- "마지막 30일" => X2[-1] => normalize => shape(60)
    last_merged_2d = X2[-1]
    last_merged_2d_norm = normalize(last_merged_2d, _mean, _std)
    
    # => sign + magnitude with scale=10^5
    arr_mag, arr_sign = to_fixed_point(last_merged_2d_norm, scale=SCALE)

    # JSON 저장
    out_data = {
      "inputMag": arr_mag,
      "inputSign": arr_sign
    }

    output_file = "input_1.json"
    with open(output_file, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved inference input to {output_file}")
    print("Vector length:", len(arr_mag))
    print("First few magnitude:", arr_mag[:5])
    print("First few sign:", arr_sign[:5])

if __name__ == '__main__':
    main()
