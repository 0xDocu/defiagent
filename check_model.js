// 파일: load_tfjs_model.js
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

async function loadModel() {
    const modelDir = 'web2_models';
    const modelPath = `file://${path.resolve(modelDir)}/model.json`;
    console.log("modelPath", modelPath);

    const model = await tf.loadLayersModel(modelPath);
    console.log("모델 로드 성공!");

    model.summary();
    
    console.log("\n모델 추가 정보:");
    console.log("Input Shape:", model.inputs[0].shape);
    console.log("Output Shape:", model.outputs[0].shape);

    const testInput = tf.zeros([1, 60]);
    const prediction = model.predict(testInput);

    console.log("\nTest Prediction:");
    prediction.print();

}

loadModel();
