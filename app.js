const express = require('express');
const multer = require('multer');
const {
    v4: uuidv4
} = require('uuid');
const tf = require('@tensorflow/tfjs-node');
const fetch = require('node-fetch');

const app = express();
const port = 3000;

async function loadModel() {
    const modelUrl = 'https://storage.googleapis.com/zzidzz-model-bucket/model.json';
    const model = await tf.loadLayersModel(modelUrl);
    return model;
}

let model;
loadModel().then(loadedModel => {
    model = loadedModel;
    console.log('Model loaded successfully');
}).catch(err => {
    console.error('Failed to load the model:', err);
});

const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 1000000
    } // 1mb
}).single('image');

app.post('/predict', (req, res) => {
    upload(req, res, async function (error) {
        if (error) {
            if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(413).json({
                    status: "fail",
                    message: "Payload content length greater than maximum allowed: 1000000"
                });
            }
            return res.status(400).json({
                status: "fail",
                message: error.message
            });
        }

        if (!req.file) {
            return res.status(400).json({
                status: "fail",
                message: "No image file provided"
            });
        }

        try {
            const imgBuffer = req.file.buffer;
            const imgTensor = tf.node.decodeImage(imgBuffer, 3);
            const resized = imgTensor.expandDims(0).toFloat().div(tf.scalar(255));
            const prediction = await model.predict(resized);

            const output = prediction.dataSync()[0];
            const result = output > 0.5 ? "Cancer" : "Non-cancer";

            const response = {
                status: "success",
                message: "Model is predicted successfully",
                data: {
                    id: uuidv4(),
                    result: result,
                    suggestion: "Consult a specialist",
                    createdAt: new Date().toISOString()
                }
            };
            res.json(response);
        } catch (err) {
            res.status(500).json({
                status: "fail",
                message: "Error in prediction"
            });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});