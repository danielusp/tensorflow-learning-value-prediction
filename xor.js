const tf = require('@tensorflow/tfjs-node');

(async () => {
    // Creating a model to predict the output
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 8,
        inputShape: 2,
        activation: 'tanh'
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        optimizer: 'sgd',
        loss: 'binaryCrossentropy',
        lr: 0.1
    });

    // Creating dataset
    const xs = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]);
    xs.print();

    const ys = tf.tensor2d([
        [0],
        [1],
        [1],
        [0]
    ]);
    ys.print();

    // Train the model
    await model.fit(xs, ys, {
        batchSize: 1,
        epochs: 5000
    });

    // Predict
    model.predict(xs).print();
})();