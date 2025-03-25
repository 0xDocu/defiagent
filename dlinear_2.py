# DLinear model
# SUIUSDT 데이터로 SUI-USDT pool의 3days average APY 예측

# target data: 'dataset/data_1st.csv'

# price만 정규화, 결과인 apys는 정규화 X
# training set 정규화 후, 해당 set의 mean, std를 사용하여 val/test set 정규화

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

DATA_CSV = 'data/dataset/data_1st.csv'

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
    trend_seq = keras.layers.AveragePooling1D(pool_size=5, strides=1, padding='same')(x)

    # 잔차 / lambda layer로 구현 (모델 복원 문제)
    resid_seq = keras.layers.Lambda(
        lambda t: t[0] - t[1]
    )([x, trend_seq])
    
    # flatten
    trend_flat = keras.layers.Flatten()(trend_seq)   # (batch, window_size)
    resid_flat = keras.layers.Flatten()(resid_seq)   # (batch, window_size)
    
    # Dense
    trend_pred = keras.layers.Dense(horizon)(trend_flat)
    resid_pred = keras.layers.Dense(horizon)(resid_flat)
    
    y_pred = keras.layers.Add()([trend_pred, resid_pred])
    
    model = keras.Model(inputs, y_pred)
    return model

def main():
    df = pd.read_csv(DATA_CSV)
    df.sort_values('date', inplace=True, ignore_index=True)
    print("df.shape =", df.shape)
    
    # csv의 apy값이 이미 3일 평균값으로 전치리된 상황이므로 horizon=1
    X, Y, date_label = create_window_multistep(df, 'price', 'apy', 30, 1)
    
    # train 70% / val 15% / test 15%
    N = len(X)
    train_end = int(N*0.7)
    val_end   = int(N*0.85)
    
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val,   Y_val   = X[train_end:val_end], Y[train_end:val_end]
    X_test,  Y_test  = X[val_end:], Y[val_end:]

    print("Train size:", X_train.shape, Y_train.shape)
    print("Val size:", X_val.shape, Y_val.shape)
    print("Test size:", X_test.shape, Y_test.shape)

    train_flat = X_train.reshape(-1, X_train.shape[-1])
    _mean = train_flat.mean(axis=0)
    _std = train_flat.std(axis=0)

    def apply_normalize(x2d, _mean, _std):
        # x2d.shape = (batch, window_size)
        x2d_norm = (x2d - _mean) / _std
        return x2d_norm

    X_train = apply_normalize(X_train, _mean, _std)
    X_val   = apply_normalize(X_val,   _mean, _std)
    X_test  = apply_normalize(X_test,  _mean, _std)

    print("Check if X_test has NaNs: ", np.isnan(X_test).any())
    print("Check if X_test has Infs: ", np.isinf(X_test).any())
    print("Check if Y_test has NaNs: ", np.isnan(Y_test).any())
    print("Check if Y_test has Infs: ", np.isinf(Y_test).any())
    
    # building model
    model = build_dlinear_multistep(30, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = tf.keras.losses.MeanSquaredError(), 
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
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
    y_pred = model.predict(test_ds)
    print("y_pred shape =", y_pred.shape)

    y_test_concat = np.concatenate([t for _, t in test_ds], axis=0)
    print("y_test shape =", y_test_concat.shape)
    
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
    plt.subplot(1,2,2)
    plt.plot(history.history['mean_absolute_error'], label='train_mae')
    plt.plot(history.history['val_mean_absolute_error'], label='val_mae')
    plt.title("MAE Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=False)  # 그래프 창을 표시하되 block=False로 설정

    # Save the model as .h5
    print(f"\nSaving model to...")
    model.save("model/dlinear_2_model.keras")
    print("Model saved successfully tf.")
    model.save("model/dlinear_2_model.h5")
    print("Model saved successfully .h5")


    print("Close the figure window or press Enter in the console to exit.")
    input("Press Enter to exit...\n") 

# python -c "import tensorflow as tf; model = tf.keras.models.load_model('model/dlinear_2_model.h5'); model.summary()" 

if __name__ == '__main__':
    main()
