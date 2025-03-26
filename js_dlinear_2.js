//ES Module
import fs from 'fs';
import { parse } from 'csv-parse/sync'; 
import * as tf from '@tensorflow/tfjs-node';

// CSV 경로
const DATA_CSV = 'data/dataset/data_1st.csv';


class SubLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }
  
  call(inputs, kwargs) {
    // Ensure we're getting a single tensor or the first input tensor
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const y = Array.isArray(inputs) && inputs.length > 1 ? inputs[1] : 0;
    return x.sub(y);
  }

  static get className() {
    return 'SubLayer';
  }
}

tf.serialization.registerClass(SubLayer);

function loadCSV(csvPath) {
  const fileContent = fs.readFileSync(csvPath, 'utf-8');
  // csv-parse로 파싱
  const records = parse(fileContent, {
    columns: true,    // 첫 행을 헤더로 사용
    skip_empty_lines: true
  });
  return records;
}

function createWindowMultistep(df, priceCol, apyCol, windowSize, horizon) {
  /**
   * df: JS 객체 배열 형태 (ex: [{date:'2023-01-01', price:100, apy:3}, {...}, ...])
   * return: { X, Y, dateLabel }
   *   - X: shape (N, windowSize)
   *   - Y: shape (N, horizon)
   */
  const N = df.length;
  const X_list = [];
  const Y_list = [];
  const date_label = [];

  for (let i = 0; i < N - windowSize - horizon + 1; i++) {
    const xWindow = [];
    const yWindow = [];
    for (let j = 0; j < windowSize; j++) {
      xWindow.push(parseFloat(df[i + j][priceCol]));
    }
    for (let h = 0; h < horizon; h++) {
      yWindow.push(parseFloat(df[i + windowSize + h][apyCol]));
    }
    // label date
    date_label.push(df[i + windowSize]['date']);
    X_list.push(xWindow);
    Y_list.push(yWindow);
  }

  // 2차원 배열을 텐서로 변환
  const X = tf.tensor2d(X_list);
  const Y = tf.tensor2d(Y_list);

  return { X, Y, dateLabel: date_label };
}

function buildDLinearMultistep(windowSize, horizon) {

  const input = tf.input({ shape: [windowSize], name: 'input_layer' });

  // (windowSize,) -> (windowSize,1)
  const reshape = tf.layers
    .reshape({ targetShape: [windowSize, 1], name: 'reshape_layer' })
    .apply(input);

  // AveragePooling1D
  const pool = tf.layers
    .averagePooling1d({ poolSize: 5, strides: 1, padding: 'same', name: 'trend_layer' })
    .apply(reshape);

  const flattenTrend = tf.layers.flatten().apply(pool);
  const flattenX = tf.layers.flatten().apply(reshape);

  // resid = flattenX - flattenTrend
  const subLayer = new SubLayer({ name: 'resid_layer' });
  const resid = subLayer.apply([flattenX, flattenTrend]);

  // trend_pred
  const trendDense = tf.layers.dense({ units: horizon, name: 'trend_pred' }).apply(flattenTrend);
  // resid_pred
  const residDense = tf.layers.dense({ units: horizon, name: 'resid_pred' }).apply(resid);

  // y_pred = trend_pred + resid_pred
  const yPred = tf.layers.add().apply([trendDense, residDense]);

  // 최종 모델
  const model = tf.model({
    inputs: input,
    outputs: yPred,
    name: 'DLinear_Multistep'
  });
  return model;
}

//---------------------------------------------
// (4) 메인 실행
//---------------------------------------------
async function main() {
  // 1) CSV 로드
  const df = loadCSV(DATA_CSV);
  // date 기준 정렬 (파이썬의 df.sort_values('date')와 유사)
  df.sort((a, b) => new Date(a.date) - new Date(b.date));
  console.log(`Data loaded. shape=${df.length}`);

  // 2) 윈도우 만들기
  const windowSize = 30;
  const horizon = 1;
  const { X, Y, dateLabel } = createWindowMultistep(df, 'price', 'apy', windowSize, horizon);
  console.log('X shape=', X.shape, 'Y shape=', Y.shape);

  // 3) train/val/test 분할
  const N = X.shape[0];
  const trainEnd = Math.floor(N * 0.7);
  const valEnd   = Math.floor(N * 0.85);

  // slice는 [start, end) 형태
  const X_train = X.slice([0, 0], [trainEnd, windowSize]);
  const Y_train = Y.slice([0, 0], [trainEnd, horizon]);

  const X_val = X.slice([trainEnd, 0], [valEnd - trainEnd, windowSize]);
  const Y_val = Y.slice([trainEnd, 0], [valEnd - trainEnd, horizon]);

  const X_test = X.slice([valEnd, 0], [N - valEnd, windowSize]);
  const Y_test = Y.slice([valEnd, 0], [N - valEnd, horizon]);

  console.log(`Train size: ${X_train.shape} -> ${Y_train.shape}`);
  console.log(`Val size:   ${X_val.shape} -> ${Y_val.shape}`);
  console.log(`Test size:  ${X_test.shape} -> ${Y_test.shape}`);

  // 4) train set으로 price 정규화
  //    (mean, std) 계산 -> 동일 값으로 val/test 정규화
  const trainFlat = X_train.reshape([trainEnd * windowSize]);
  const meanTensor = trainFlat.mean();
  const stdTensor  = trainFlat.sub(meanTensor).square().mean().sqrt(); // 표준편차

  function applyNormalize(tensor) {
    return tensor.sub(meanTensor).div(stdTensor);
  }

  const X_train_norm = applyNormalize(X_train);
  const X_val_norm   = applyNormalize(X_val);
  const X_test_norm  = applyNormalize(X_test);

  // 5) 모델 구성
  const model = buildDLinearMultistep(windowSize, horizon);
  model.compile({
    optimizer: tf.train.adam(1e-4),
    loss: tf.losses.meanSquaredError,
    metrics: [tf.metrics.meanAbsoluteError]
  });
  model.summary();

  // 6) tfjs의 Dataset 구성 (여기서는 tf.data.Dataset.fromTensorSlices() + batch)
  //    Node 환경에서는 fitDataset 대신 fit 사용도 가능
  const batchSize = 32;

  const trainDataset = tf.data.zip({
    xs: tf.data.array(X_train_norm.arraySync()),
    ys: tf.data.array(Y_train.arraySync())
  }).batch(batchSize);
  
  const valDataset = tf.data.zip({
    xs: tf.data.array(X_val_norm.arraySync()),
    ys: tf.data.array(Y_val.arraySync())
  }).batch(batchSize);
  
  const testDataset = tf.data.zip({
    xs: tf.data.array(X_test_norm.arraySync()),
    ys: tf.data.array(Y_test.arraySync())
  }).batch(batchSize);

  // EarlyStopping 등 콜백
  const earlyStop = tf.callbacks.earlyStopping({
    monitor: 'val_loss',
    patience: 10,
  });

  // 7) 훈련
  console.log('\nStarting model training...');
  const history = await model.fitDataset(trainDataset, {
    epochs: 200,
    validationData: valDataset,
    callbacks: [earlyStop],
    verbose: 1
  });

  const evalResult = await model.evaluateDataset(testDataset);
  const mse = evalResult[0].dataSync()[0].toFixed(4);
  const mae = evalResult[1].dataSync()[0].toFixed(4);
  console.log(`[Test] MSE=${mse}, MAE=${mae}`);

  // 9) 예측
  const predsArray = [];
  await testDataset.forEachAsync(batch => {
    // batch.xs는 입력 텐서입니다.
    const predBatch = model.predict(batch.xs);
    // predBatch는 텐서이므로 arraySync()를 호출하여 자바스크립트 배열로 변환
    predsArray.push(...predBatch.arraySync());
  });
  console.log('Prediction shape:', predsArray.length, '(each row = horizon=1)');

  // 10) 모델 저장
  //     node 환경에서 로컬폴더에 저장 시: 'file://상대경로' 형태 필수
  //     ex) model.save('file://my_model')
  await model.save('file://model/dlinear_2_model_tfjs');
  console.log('Model saved at model/dlinear_2_model_tfjs');

  // (옵션) 혹은 tfjs 라이브러리에서 h5와 유사하게?
  //  - tfjs는 HDF5 형태로 직접 저장 불가능. 
  //  - "file://" prefix 폴더 구조에 model.json + weights.bin 생성
  //  - 이후 browser용 변환 필요 시에는 tfjs-converter를 다시 쓸 수도 있습니다.

  tf.disposeVariables();
  console.log('Done!');
}

// 실행
main().catch((err) => console.error(err));
