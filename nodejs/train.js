
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const { exit } = require('process')
const BertWordPieceTokenizer = require('tokenizers').BertWordPieceTokenizer


function buildModel() {
    const input = tf.input({shape: [null, 768], dtype: 'float32'})
    const rnn = tf.layers.bidirectional({
        layer: tf.layers.lstm({units: 128, returnSequences: false})
    })
    const mask = tf.layers.masking({maskValue: 0.0})
    const dense = tf.layers.dense({units: 2, activation: 'softmax'})
    const output = dense.apply(rnn.apply(mask.apply(input)))
    const model = tf.model({inputs: input, outputs: output})
    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['acc'],
    })
    return model
}


(async () => {

    const wordPieceTokenizer = await BertWordPieceTokenizer.fromOptions({ vocabFile: "./vocab.txt" })
    const bert = await tf.node.loadSavedModel('./bert')

    function makeGenerator(objs, batchSize) {
        function* dataGenerator() {
            let xs = []
            let ys = []
            for (const obj of objs) {
                xs.push(obj['tokens'])
                ys.push(Number.parseInt(obj['label']))
                if (xs.length == ys.length && xs.length == batchSize) {
                    const maxLength = Math.max.apply(
                        Math,
                        xs.map(x => x.length)
                    )
                    xs = xs.map(x => {
                        while(x.length < maxLength) {
                            x = x.concat([''])
                        }
                        return x
                    })
                    xs = tf.tensor(xs)
                    xs = bert.predict({
                        input_1: xs
                    })['sequence_output']
                    ys = tf.tensor(ys)
                    yield {xs, ys}
                    xs = []
                    ys = []
                }
            }
        }
        return dataGenerator
    }

    console.log('Read dataset')
    const trainObjs = fs.readFileSync(
        'train.json',
        {encoding: 'utf-8'}
    ).split(/\n/).map(JSON.parse)
    const devObjs = fs.readFileSync(
        'dev.json',
        {encoding: 'utf-8'}
    ).split(/\n/).map(JSON.parse)

    console.log('Tokenize train dataset')
    for (const obj of trainObjs) {
        obj['tokens'] = (await wordPieceTokenizer.encode(
            obj['sentence1'], obj['sentence2']
        )).tokens
    }
    console.log('Tokenize dev dataset')
    for (const obj of devObjs) {
        obj['tokens'] = (await wordPieceTokenizer.encode(
            obj['sentence1'], obj['sentence2']
        )).tokens
    }
    console.log('Start training')
    
    const batchSize = 32
    const dsTrain = tf.data.generator(makeGenerator(trainObjs, batchSize)).repeat()
    const dsDev = tf.data.generator(makeGenerator(devObjs, batchSize)).repeat()
    const model = buildModel()
    model.fitDataset(dsTrain, {
        batchesPerEpoch: Math.floor(trainObjs.length / batchSize),
        epochs: 1,
        batch_size: batchSize,
        validationData: dsDev,
        validationBatches: Math.floor(devObjs.length / batchSize),
    })

    model.evaluateDataset(dsDev, {
        batches: Math.floor(devObjs.length / batchSize),
    })

})()
