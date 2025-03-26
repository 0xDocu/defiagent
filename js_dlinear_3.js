//ES Modules
import fs from 'fs';
import { parse } from 'csv-parse/sync';
import * as tf from '@tensorflow/tfjs-node';
import promptSync from 'prompt-sync';

const DATA_CSV = 'data/dataset/data_1st.csv';
const WINDOW_SIZE = 30;
const HORIZON = 1;

// offchainPreprocess: 5일 이동평균(추세)와 잔차 계산 (SAME padding 방식)
function offchainPreprocess(prices) {
  const windowSize = prices.length;
  const trend = new Array(windowSize).fill(0);
  for (let i = 0; i < windowSize; i++) {
    const left = Math.max(0, i - 2);
    const right = Math.min(windowSize, i + 3);
    const seg = prices.slice(left, right);
    const avg = seg.reduce((sum, v) => sum + v, 0) / seg.length;
    trend[i] = avg;
  }
  const resid = prices.map((v, i) => v - trend[i]);
  return { trend, resid };
}

// CSV 데이터에서 윈도우와 레이블을 구성 (가격, apy, 날짜)
function createWindowMultistep(df, priceCol, apyCol, windowSize, horizon) {
  const X_list = [];
  const Y_list = [];
  const dateLabel = [];
  const N = df.length;
  for (let i = 0; i < N - windowSize - horizon + 1; i++) {
    const xWindow = [];
    const yWindow = [];
    for (let j = 0; j < windowSize; j++) {
      xWindow.push(parseFloat(df[i + j][priceCol]));
    }
    for (let h = 0; h < horizon; h++) {
      yWindow.push(parseFloat(df[i + windowSize + h][apyCol]));
    }
    dateLabel.push(df[i + windowSize]['date']);
    X_list.push(xWindow);
    Y_list.push(yWindow);
  }
  return { X: X_list, Y: Y_list, dateLabel };
}

// 3-layer MLP 모델 생성 (입력: flatten된 [trend, resid] → 60차원)
function buildLightMLP(inputDim) {
  const input = tf.input({ shape: [inputDim], name: 'input_layer' });
  const dense1 = tf.layers.dense({ units: 32, activation: 'relu', name: 'dense' }).apply(input);
  const dense2 = tf.layers.dense({ units: 16, activation: 'relu', name: 'dense_1' }).apply(dense1);
  const output = tf.layers.dense({ units: 1, name: 'dense_2' }).apply(dense2);
  return tf.model({ inputs: input, outputs: output, name: '3layer_mlp' });
}

async function main() {
  // CSV 파일 로드
  const csvData = fs.readFileSync(DATA_CSV, 'utf-8');
  const records = parse(csvData, { columns: true, skip_empty_lines: true });
  // 날짜 기준 정렬
  records.sort((a, b) => new Date(a.date) - new Date(b.date));
  console.log("df.shape =", records.length);
  
  // 윈도우 생성
  const { X: X_raw, Y, dateLabel } = createWindowMultistep(records, 'price', 'apy', WINDOW_SIZE, HORIZON);
  console.log("Initial X_raw shape =", [X_raw.length, WINDOW_SIZE]);
  console.log("Y shape =", [Y.length, HORIZON]);
  
  // 각 윈도우에 대해 offchainPreprocess를 적용한 후, trend와 resid를 합쳐 60차원 입력 생성
  const X2 = X_raw.map(window => {
    const prices = window.map(parseFloat);
    const { trend, resid } = offchainPreprocess(prices);
    return trend.concat(resid);
  });
  console.log("X2 shape =", [X2.length, WINDOW_SIZE * 2]);
  
  // 데이터 분할 (70% train, 15% val, 15% test)
  const N = X2.length;
  const trainEnd = Math.floor(N * 0.7);
  const valEnd = Math.floor(N * 0.85);
  const X2_train = X2.slice(0, trainEnd);
  const Y_train = Y.slice(0, trainEnd);
  const X2_val = X2.slice(trainEnd, valEnd);
  const Y_val = Y.slice(trainEnd, valEnd);
  const X2_test = X2.slice(valEnd);
  const Y_test = Y.slice(valEnd);
  
  console.log("Train size:", [X2_train.length, WINDOW_SIZE * 2], Y_train.length);
  console.log("Val size:", [X2_val.length, WINDOW_SIZE * 2], Y_val.length);
  console.log("Test size:", [X2_test.length, WINDOW_SIZE * 2], Y_test.length);
  
  // 텐서 변환
  const X2_train_tensor = tf.tensor2d(X2_train);
  const X2_val_tensor = tf.tensor2d(X2_val);
  const X2_test_tensor = tf.tensor2d(X2_test);
  
  // 정규화: training set의 평균과 표준편차를 구함
  const trainMean = X2_train_tensor.mean(0);
  const trainStd = tf.moments(X2_train_tensor, 0).variance.sqrt().add(1e-8);
  
  const normalize = (tensor, mean, std) => tensor.sub(mean).div(std);
  
  const X2_train_norm = normalize(X2_train_tensor, trainMean, trainStd);
  const X2_val_norm = normalize(X2_val_tensor, trainMean, trainStd);
  const X2_test_norm = normalize(X2_test_tensor, trainMean, trainStd);
  
  // 모델 생성 및 컴파일
  const model = buildLightMLP(WINDOW_SIZE * 2);
  model.compile({
    optimizer: tf.train.adam(1e-4),
    loss: tf.losses.meanSquaredError,
    metrics: [tf.metrics.meanAbsoluteError]
  });
  model.summary();
  
  // tf.data.Dataset 구성 (배치 사이즈 32)
  const batchSize = 32;
  const Y_train_tensor = tf.tensor2d(Y_train);
  const Y_val_tensor = tf.tensor2d(Y_val);
  
  const trainDataset = tf.data.zip({
    xs: tf.data.array(X2_train_norm.arraySync()),
    ys: tf.data.array(Y_train_tensor.arraySync())
  }).batch(batchSize);
  
  const valDataset = tf.data.zip({
    xs: tf.data.array(X2_val_norm.arraySync()),
    ys: tf.data.array(Y_val_tensor.arraySync())
  }).batch(batchSize);
  
  console.log("Starting model training...");
  const history = await model.fitDataset(trainDataset, {
    epochs: 200,
    validationData: valDataset,
    callbacks: [tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })],
    verbose: 1
  });
  
  // 평가
  const Y_test_tensor = tf.tensor2d(Y_test);
  const testDataset = tf.data.zip({
    xs: tf.data.array(X2_test_norm.arraySync()),
    ys: tf.data.array(Y_test_tensor.arraySync())
  }).batch(batchSize);
  
  const evalResult = await model.evaluateDataset(testDataset);
  const mse = evalResult[0].dataSync()[0].toFixed(4);
  const mae = evalResult[1].dataSync()[0].toFixed(4);
  console.log(`[Test] MSE=${mse}, MAE=${mae}`);
  
  // 예측
  const predsArray = [];
  await testDataset.forEachAsync(batch => {
    const predBatch = model.predict(batch.xs);
    predsArray.push(...predBatch.arraySync());
  });
  console.log("Prediction shape:", predsArray.length);
  
  // 모델 저장 (TFJS 형식)
  await model.save('file://model/dlinear_3_model_tfjs');
  console.log("Model saved successfully at model/dlinear_3_model_tfjs");
  
  // 종료 대기 (prompt)
  const prompt = promptSync();
  prompt("Press Enter to exit...\n");
  
  tf.disposeVariables();
}

main().catch(err => console.error(err));
