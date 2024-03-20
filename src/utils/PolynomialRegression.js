import * as tf from "@tensorflow/tfjs";

export default class PolynomialRegression {
  /**
   * Calculates the mean value from a given vector.
   *
   * @param {number[]} vector - The vector containing numeric values.
   * @returns {number} - The mean value from the vector.
   */
  meanFromVector = (vector) =>
    vector.reduce((acc, val) => acc + val, 0) / vector.length;

  sdFromVector = (vector, vectorMean) =>
    Math.sqrt(
      vector.map((x) => Math.pow(x - vectorMean, 2)).reduce((a, b) => a + b) /
        vector.length -
        1,
    );

  normVector = (vector, vectorMean, vectorStdDev) =>
    vector.map((x) => (x - vectorMean) / vectorStdDev);

  prepareNormalizedTensors = (pairData, polynomialDegree) => {
    const batchSize = pairData.length;

    const xData = [];
    const yData = [];
    pairData.forEach((pair) => {
      xData.push(pair[0]);
      yData.push(pair[1]);
    });

    const meanY = this.meanFromVector(yData);
    const stddevY = this.sdFromVector(yData, meanY);
    const yNormalized = this.normVector(yData, meanY, stddevY);

    const normalizedXPowers = [];
    const xPowerMeans = [];
    const xPowerStddevs = [];

    for (let i = 0; i < polynomialDegree; ++i) {
      const xPower = xData.map((element) => Math.pow(element, i + 1));

      const meanXPower = this.meanFromVector(xPower);
      xPowerMeans.push(meanXPower);

      const stddevXPower = this.sdFromVector(xPower, meanXPower);
      xPowerStddevs.push(stddevXPower);

      const normalizedXPower = this.normVector(
        xPower,
        meanXPower,
        stddevXPower,
      );
      normalizedXPowers.push(normalizedXPower);
    }

    const xArrayData = normalizedXPowers.reduce(
      (acc, item) => [...acc, ...item],
      [1],
    );

    return [
      xPowerMeans,
      xPowerStddevs,
      tf.tensor2d(xArrayData, [batchSize, polynomialDegree + 1]),
      meanY,
      stddevY,
      tf.tensor2d(yNormalized, [batchSize, 1]),
    ];
  };

  fitModel = async (
    pairData,
    numberOfEpochs,
    learningRate,
    polynomialDegree,
  ) => {
    const batchSize = pairData.length;
    const outputs = this.prepareNormalizedTensors(pairData, polynomialDegree);

    const [
      xPowerMeans,
      xPowerStddevs,
      tensorXData,
      meanY,
      stddevY,
      tensorYData,
    ] = outputs;

    const input = tf.input({ shape: [polynomialDegree + 1] });
    const linearLayer = tf.layers.dense({
      units: 1,
      kernelInitializer: "Zeros",
      useBias: false,
    });
    const output = linearLayer.apply(input);
    const model = tf.model({ inputs: input, outputs: output });

    const sgd = tf.train.sgd(learningRate);
    model.compile({ optimizer: sgd, loss: "meanSquaredError" });

    await model.fit(tensorXData, tensorYData, {
      batchSize: batchSize,
      epochs: numberOfEpochs,
    });

    console.log(
      "Model weights (normalized):",
      model.trainableWeights[0].read().dataSync(),
    );

    return [model, xPowerMeans, xPowerStddevs, meanY, stddevY];
  };
}
