const tf = require('@tensorflow/tfjs-node');

(async () => {
    // Creating a model to predict the output
    const model = tf.sequential();
   model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    // Creating dataset
    const xs = tf.tensor1d([1,2,3,4,5,6,7,8,9,10], 'int32');
    const ys = tf.tensor1d([4,6,8,10,12,14,16,18,20,22], 'int32');

    // Train the model
    await model.fit(xs, ys, {
        batchSize: 1,
        epochs: 100
    });

    /**
     * Predict
     * 
     * In this case tensor2d(values, shape, dataType)
     * value    -> number to predict
     * shape    -> the tensor shape. in this case 1 x 1 or a single number
     * dataType -> may be float32, int32 or bool. In this case is a integer int32
     * 
     */
    model.predict(tf.tensor2d([24],[1,1]), 'int32').print();
})();