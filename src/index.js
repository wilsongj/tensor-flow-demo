import PolynomialRegression from "./utils/PolynomialRegression";
import CanvasDrawer from "./utils/CanvasDrawer";

const polyReg = new PolynomialRegression();
const drawer = new CanvasDrawer(document.getElementById("canvas"));
const order = 3;
const cubicCoeffElement = document.getElementById("cubic-coeff");
const quadCoeffElement = document.getElementById("quad-coeff");
const linearCoeffElement = document.getElementById("linear-coeff");
const constCoeffElement = document.getElementById("const-coeff");

const epochsElement = document.getElementById("epochs");
const learningRateElement = document.getElementById("learning-rate");

const calculateXPowers = (x, order, xPowerMeans, xPowerStddevs) => {
  let d = 1;
  const xPowers = [];
  for (let j = 0; j < order + 1; ++j) {
    xPowers.push(j === 0 ? d : (d - xPowerMeans[j - 1]) / xPowerStddevs[j - 1]);
    d *= x;
  }
  return xPowers;
};

const renderModelPredictions = (
  order,
  model,
  xPowerMeans,
  xPowerStddevs,
  yMean,
  yStddev,
) => {
  const ctx = drawer.canvas.getContext("2d");
  const width = drawer.canvas.width;
  const xStep = 0.02 * width;
  const xs = [];
  const xPowers = [];
  for (let x = -0.5 * width; x < 0.5 * width; x += xStep) {
    xs.push(x);
    xPowers.push(...calculateXPowers(x, order, xPowerMeans, xPowerStddevs));
  }

  const predictOut = model.predict(
    tf.tensor2d(xPowers, [xs.length, order + 1]),
  );
  const normalizedYs = predictOut.dataSync();
  ctx.beginPath();
  let canvasXY = drawer.convertWorldCoordsToCanvas(
    xs[0],
    normalizedYs[0] * yStddev + yMean,
  );
  ctx.moveTo(...canvasXY);
  for (let i = 1; i < xs.length; ++i) {
    canvasXY = drawer.convertWorldCoordsToCanvas(
      xs[i],
      normalizedYs[i] * yStddev + yMean,
    );
    ctx.lineTo(...canvasXY);
    ctx.stroke();
  }
};

const calculatePolynomial = (x, coeffs) =>
  coeffs[0] * Math.pow(x, 3) +
  coeffs[1] * Math.pow(x, 2) +
  coeffs[2] * x +
  coeffs[3];

const generateXYData = (coeffs) => {
  const numPoints = drawer.canvas.width / 25;
  return Array.from({ length: numPoints }, (_, i) => {
    const x = i - drawer.canvas.width / 2;
    const y = calculatePolynomial(x, coeffs);
    return [x, y];
  });
};

const fitAndRender = async () => {
  const epochs = +epochsElement.value;
  const learningRate = +learningRateElement.value;
  if (!isFinite(epochs) || !isFinite(learningRate)) {
    return;
  }

  const [cubicCoeff, quadCoeff, linearCoeff, constCoeff] = [
    +cubicCoeffElement.value,
    +quadCoeffElement.value,
    +linearCoeffElement.value,
    +constCoeffElement.value,
  ];
  console.log(
    "True coefficients: " +
      JSON.stringify({ cubicCoeff, quadCoeff, linearCoeff, constCoeff }),
  );

  const xyData = generateXYData({
    cubicCoeff,
    quadCoeff,
    linearCoeff,
    constCoeff,
  });
  drawer.drawXYData(xyData);

  const [model, xPowerMeans, xPowerStddevs, yMean, yStddev] =
    await polyReg.fitModel(xyData, epochs, learningRate);

  await renderModelPredictions(
    model,
    xPowerMeans,
    xPowerStddevs,
    yMean,
    yStddev,
  );
};

cubicCoeffElement.addEventListener("change", fitAndRender);
quadCoeffElement.addEventListener("change", fitAndRender);
linearCoeffElement.addEventListener("change", fitAndRender);
constCoeffElement.addEventListener("change", fitAndRender);
epochsElement.addEventListener("change", fitAndRender);
learningRateElement.addEventListener("change", fitAndRender);

fitAndRender().catch(console.error);
