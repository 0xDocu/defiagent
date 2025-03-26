// 파일: load_tfjs_model.js
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

class SubLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
    }
    computeOutputShape(inputShape) {
        return inputShape[0];
    }
    call(inputs, kwargs) {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        const y = Array.isArray(inputs) && inputs.length > 1 ? inputs[1] : 0;
        return x.sub(y);
    }
    static get className() {
        return 'SubLayer';
    }
}
tf.serialization.registerClass(SubLayer);

async function loadModel() {
    const modelDir = 'TFJS_model';
    const modelPath = `file://${path.resolve(modelDir)}/model.json`;
    console.log("modelPath", modelPath);

    const localModelJsonPath = path.resolve(modelDir, 'model.json');
    if (!fs.existsSync(localModelJsonPath)) {
        console.error(`model.json 파일이 '${localModelJsonPath}'에 존재하지 않습니다.`);
        return;
    } else {
        console.log(`model.json 파일이 '${localModelJsonPath}'에 정상적으로 존재합니다.`);
    }

    try {
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
    
    } catch (err) {
        console.error("모델 로딩 중 오류:", err);
        return;
    }
   
    

}

loadModel();
