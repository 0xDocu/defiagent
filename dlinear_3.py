# dlinear_3.py
# dlinear_2에서 DLinear model은 다음 layer를 사용
'''
   1. AveragePooling1D: 추세 추출
   2. Lambda( x - trend ): 잔차 계산
   3. Flatten 2회 (추세/잔차 각각)
   4. Dense 2회 (추세/잔차 예측)
   5. Lambda( trend_pred + resid_pred ): 최종 합산
'''
# 이를 tensorflowSui를 통해 온체인에 업로드하고, 온체인에서 연산을 수행하기 위해서는 가스비 등 부하를 고려하여 input layer 포함 4개 layer 이하로 변경 필요
# average pooling과 residual 계산은 방법을 고지하고, 사용자의 input에 대해 intermediate 값을 공개하면 투명성 문제 해결

# target data: 'dataset/data_1st.csv'
# price만 정규화, 결과인 apys는 정규화 X
# training set 정규화 후, 해당 set의 mean, std를 사용하여 val/test set 정규화

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

DATA_CSV = 'data/dataset/data_1st.csv'
WINDOW_SIZE = 30
HORIZON = 1

# 전처리 함수, 5일 이평선과 (종가-5일 이평)의 잔차 계산
def offchain_preprocess(prices_1d):
    # input: np.array of shape (window_size,) (ex: length 30)
    # output: (trend_flat, resid_flat) both shape=(window_size,)
    window_size = len(prices_1d)

    # average pooling with pool_size=5 'SAME' padding
    trend_seq = np.zeros(window_size)
    for i in range(window_size):
        # 'SAME' padding => index range = i-2..i+2
        # handle boundary carefully
        left = max(0, i-2)
        right= min(window_size, i+3)
        seg = prices_1d[left:right]
        trend_seq[i] = np.mean(seg)
    resid_seq = prices_1d - trend_seq
    return trend_seq, resid_seq

# 전처리 후 (trend, resid) 합쳐서 모델에 입력으로 전송
# 3 layer
def build_light_mlp(input_dim):
    # input_dim = 2 * window_size if we flatten (trend, resid) together (or just window_size if we do something else)
    inputs = keras.Input(shape=(input_dim,), name='input_layer')
    
    x = keras.layers.Dense(32, activation='relu', name='dense')(inputs)

    x = keras.layers.Dense(16, activation='relu', name='dense_1')(x)
    
    outputs = keras.layers.Dense(1, name='dense_2')(x)

    model = keras.Model(inputs, outputs, name='3layer_mlp')
    return model

# 2 layer
def build_2layer_mlp(input_dim):
    """
    - Layer1 ("dense"): 32 units (ReLU)
    - Layer2 ("dense_1"): 1 unit (Linear output)
    """
    inputs = keras.Input(shape=(input_dim,), name='input_layer')
    
    x = keras.layers.Dense(32, activation='relu', name='dense')(inputs)
    
    outputs = keras.layers.Dense(1, name='dense_1')(x)
    
    model = keras.Model(inputs, outputs, name='2layer_mlp')
    return model

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

def convert_model_to_tfjs(input_path, output_path):
    try:
        inputs = tf.keras.Input(shape=(60,), name='input_layer')
        x = tf.keras.layers.Dense(32, activation='relu', name='dense')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu', name='dense_1')(x)
        outputs = tf.keras.layers.Dense(1, name='dense_2')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        # 저장된 가중치 로드
        model.load_weights(input_path)
        
        # TFJS로 변환
        tfjs.converters.save_keras_model( model, output_path )
        
        print(f"Model successfully converted to TFJS and saved at {output_path}")

    except Exception as e:
        print(f"Conversion error: {e}")


def main():
    df = pd.read_csv(DATA_CSV)
    df.sort_values('date', inplace=True, ignore_index=True)
    print("df.shape =", df.shape)
    
    # csv의 apy값이 이미 3일 평균값으로 전치리된 상황이므로 horizon=1
    X_raw, Y, date_label = create_window_multistep(df, 'price', 'apy', WINDOW_SIZE, HORIZON)
    print("Initial X_raw shape =", X_raw.shape)   # (batch, 30)
    print("Y shape =", Y.shape)                  # (batch, 1)

    # X, Y 로 윈도우를 뽑되, AveragePooling + resid는 PYTHON에서 미리 처리
    X2_list = []
    for x_window in X_raw:
        trend_seq, resid_seq = offchain_preprocess(x_window)
        # concate to shape (30*2,)
        merged_2d = np.concatenate([trend_seq, resid_seq], axis=0)
        X2_list.append(merged_2d)

    X2 = np.array(X2_list)  # shape=(batch, 60) if window_size=30
    print("X2 shape =", X2.shape)
    
    # train 70% / val 15% / test 15%
    N = len(X2)
    train_end = int(N*0.7)
    val_end   = int(N*0.85)
    
    X2_train, Y_train = X2[:train_end], Y[:train_end]
    X2_val,   Y_val   = X2[train_end:val_end], Y[train_end:val_end]
    X2_test,  Y_test  = X2[val_end:], Y[val_end:]

    print("Train size:", X2_train.shape, Y_train.shape)
    print("Val size:", X2_val.shape, Y_val.shape)
    print("Test size:", X2_test.shape, Y_test.shape)

    _mean = X2_train.mean(axis=0)
    _std = X2_train.std(axis=0)

    def normalize(x, m, s):
        return (x - m) / s

    X2_train_norm = normalize(X2_train, _mean, _std)
    X2_val_norm   = normalize(X2_val,   _mean, _std)
    X2_test_norm  = normalize(X2_test,  _mean, _std)
    
    # building model
    model = build_light_mlp(WINDOW_SIZE*2)
    #model = build_2layer_mlp(WINDOW_SIZE*2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = tf.keras.losses.MeanSquaredError(), 
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X2_train_norm, Y_train)).batch(32).prefetch(1)
    val_ds   = tf.data.Dataset.from_tensor_slices((X2_val_norm,   Y_val)).batch(32).prefetch(1)

    print("== Summary of Simple Model ==")
    model.summary()
    
    # model fitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    print("Starting model training...")
    history = model.fit(train_ds, epochs=200, validation_data=val_ds, callbacks=callbacks, verbose=1)
    
    # evaluating
    print("Evaluating on test set...")
    test_ds = tf.data.Dataset.from_tensor_slices((X2_test_norm, Y_test)).batch(32).prefetch(1)
    loss, mae = model.evaluate(test_ds, verbose=0)
    print(f"[Test] MSE={loss:.4f}, MAE={mae:.4f}")
    
    # predicting
    print("Generating predictions...")
    y_pred = model.predict(test_ds)
    print("y_pred shape =", y_pred.shape)

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
    print(f"\nSaving model...")
    # .h5, TFJS 모두 저장 진행
    model.save("model/dlinear_3_model.h5")
    convert_model_to_tfjs("model/dlinear_3_model.h5", "web2_models")
    print("Model saved successfully.")

    print("Close the figure window or press Enter in the console to exit.")
    input("Press Enter to exit...\n") 
    
if __name__ == '__main__':
    main()
