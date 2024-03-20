export default class CanvasDrawer {
  constructor(canvas) {
    this.canvas = canvas;
  }

  convertWorldCoordsToCanvas = (x, y) => [
    x + this.canvas.width / 2,
    -y + this.canvas.height / 2,
  ];

  drawAxes = () => {
    const drawLine = (context, startPoint, endPoint) => {
      context.moveTo(startPoint[0], startPoint[1]);
      context.lineTo(endPoint[0], endPoint[1]);
      context.stroke();
    };
    const ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.beginPath();
    const leftCoord = this.convertWorldCoordsToCanvas(
      this.canvas,
      -this.canvas.width / 2,
      0,
    );
    const rightCoord = this.convertWorldCoordsToCanvas(
      this.canvas,
      this.canvas.width / 2,
      0,
    );
    drawLine(ctx, leftCoord, rightCoord);
    const topCoord = this.convertWorldCoordsToCanvas(
      this.canvas,
      0,
      this.canvas.height / 2,
    );
    const bottomCoord = this.convertWorldCoordsToCanvas(
      this.canvas,
      0,
      -this.canvas.height / 2,
    );
    drawLine(ctx, topCoord, bottomCoord);
  };

  drawXYData = (xyData) => {
    this.drawAxes();
    const plotPoint = (context, [x, y]) => {
      context.beginPath();
      context.arc(x, y, 4, 0, Math.PI * 2, true);
      context.stroke();
    };
    const ctx = this.canvas.getContext("2d");
    xyData.forEach((xyPoint) => {
      const canvasCoord = this.convertWorldCoordsToCanvas(...xyPoint);
      plotPoint(ctx, canvasCoord);
    });
  };
}
