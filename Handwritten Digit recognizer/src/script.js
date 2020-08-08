import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data.js';

function getModel(){
    const model = tf.sequential();
    const IMG_WIDTH = 28;
    const IMG_HEIGHT = 28;
    const IMG_CHANNELS = 1;
    model.add(tf.layers.conv2d({
        inputShape: [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

  return model;
}

async function train(model,data){
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 55000;
    const TEST_DATA_SIZE = 10000;
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });
    
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });
    
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
    });
}

async function run(){
    const data = new MnistData();
    await data.load();
    const model = getModel();
    console.log("Training Model...");
    await train(model,data);
    console.log("Model trained");
    const savestatus = await model.save('localstorage://my-model');
    console.log(savestatus)
}
export default run;



