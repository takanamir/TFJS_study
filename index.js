import * as tf from '@tensorflow/tfjs';

// 線形回帰モデルを定義
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// 訓練用の模擬データを生成
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// データを使用してモデルを訓練
model.fit(xs, ys, {epochs: 10}).then(() => {
  // モデルを使用してモデルが見たことのないデータポイントを推論
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // 結果を確認するためにブラウザのDevToolsを開く
});
