import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

### Price -> APY 예측

PRICE_CSV = 'data/normalization/norm_hist_price.csv'
APY_CSV    = 'data/normalization/norm_apy_tvl.csv'

### 역정규화 정보
APY_MEAN = 144.4370
APY_STD = 115.9434

def load_and_merge():
    price_df = pd.read_csv(PRICE_CSV)
    apy_df   = pd.read_csv(APY_CSV)
    
    price_df.sort_values('date', inplace=True)
    apy_df.sort_values('date', inplace=True)

    print("load_and_merge() - price_df.shape =", price_df.shape)
    print("load_and_merge() - apy_df.shape =", apy_df.shape)
    
    df = pd.merge(price_df, apy_df, on='date', how='inner')
   
    return df

def create_window_multistep(df, price_col, apy_col, window_size, horizon):

    values_price = df[price_col].values
    values_apy   = df[apy_col].values
    dates        = df['date'].values
    
    X_list, Y_list, date_label = [], [], []
    N = len(df)
    
    for i in range(N - window_size - horizon + 1):
        x_window = values_price[i : i+window_size]
        y_window = values_apy[i+window_size : i+window_size+horizon]
        label_date = dates[i+window_size]  # y_window 시작 날짜
        
        X_list.append(x_window)
        Y_list.append(y_window)
        date_label.append(label_date)
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    date_label = np.array(date_label)
    
    return X, Y, date_label

def build_dlinear_multistep(window_size, horizon):

    inputs = keras.Input(shape=(window_size,))
    
    x = keras.layers.Reshape((window_size, 1))(inputs)
    
    # 추세
    trend_seq = keras.layers.AveragePooling1D(pool_size=25, strides=1, padding='same')(x)

    # 잔차
    resid_seq = x - trend_seq
    
    # flatten
    trend_flat = keras.layers.Flatten()(trend_seq)   # (batch, window_size)
    resid_flat = keras.layers.Flatten()(resid_seq)   # (batch, window_size)
    
    # Dense
    trend_pred = keras.layers.Dense(horizon)(trend_flat)    # shape=(batch,7)
    resid_pred = keras.layers.Dense(horizon)(resid_flat)    # shape=(batch,7)
    
    y_pred = trend_pred + resid_pred  # shape=(batch,7)
    
    model = keras.Model(inputs, y_pred)
    return model

def main():
    merged_df = load_and_merge()
    merged_df.sort_values('date', inplace=True, ignore_index=True)
    print("merged_df.shape =", merged_df.shape)
    
    #X, Y, date_label = create_window_multistep(merged_df, 'norm_price', 'merged_apy_scaled', 30, 7)
    X, Y, date_label = create_window_multistep(merged_df, 'norm_price', 'merged_apy_scaled', 30, 1)
    
    # train 70% / val 15% / test 15%
    N = len(X)
    train_end = int(N*0.7)
    val_end   = int(N*0.85)
    
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val,   Y_val   = X[train_end:val_end], Y[train_end:val_end]
    X_test,  Y_test  = X[val_end:], Y[val_end:]
    
    # building model
    model = build_dlinear_multistep(30, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse', 
        metrics=['mae']
    )
    
    # tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32).prefetch(1)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   Y_val)).batch(32).prefetch(1)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(32).prefetch(1)
    
    # check data
    print("X.shape =", X.shape)  # (전체 샘플 수, 30) ?
    print("X_train.shape =", X_train.shape)

    # model fitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=0, restore_best_weights=True)
    ]
    print("Starting model training...")
    history = model.fit(train_ds, epochs=200, validation_data=val_ds, callbacks=callbacks, verbose=1)
    
    # evaluating
    print("Evaluating on test set...")
    loss, mae_scaled = model.evaluate(test_ds)
    print(f"[Test] MSE={loss:.4f}, MAE={mae_scaled:.4f}")
    
    # predicting
    print("Generating predictions...")
    y_pred_scaled = model.predict(test_ds)  # shape=(len(X_test),7)
    print("y_pred shape =", y_pred_scaled.shape)

    # 역정규화하여 결과 분석
    y_test_scaled = np.concatenate([y for x, y in test_ds], axis=0)
    print("y_test_scaled shape =", y_test_scaled.shape)
    
    y_pred = y_pred_scaled * APY_STD + APY_MEAN
    y_test = y_test_scaled * APY_STD + APY_MEAN

    # MSE, MAE 재계산
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"[Test - original scale] MSE={mse:.4f}, MAE={mae:.4f}")

    plt.figure(figsize=(10,4))

    # 손실(loss) 그래프
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Loss Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    
    # MAE 그래프
    if 'mae' in history.history:
        plt.subplot(1,2,2)
        plt.plot(history.history['mae'], label='train_mae')
        plt.plot(history.history['val_mae'], label='val_mae')
        plt.title("MAE Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
    
    plt.tight_layout()
    plt.show(block=False)  # 그래프 창을 표시하되 block=False로 설정

    print("Close the figure window or press Enter in the console to exit.")
    input("Press Enter to exit...\n") 
    
if __name__ == '__main__':
    main()
