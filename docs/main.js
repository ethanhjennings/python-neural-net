"use strict";

var mouseDown = false;
var prevX, prevY;

var drawingCanvas;
var drawingCtx;
var scaledCanvas;
var scaledCtx;
var pixelCanvas;
var pixelCtx;
var networkCanvas;
var networkCtx;
var outputText;
var drawHereTextShowing;
var synapseDrawingIterator;

window.addEventListener("DOMContentLoaded", () => {
  drawingCanvas = document.getElementById("drawingCanvas");
  drawingCtx = drawingCanvas.getContext("2d");

  scaledCanvas = document.createElement("canvas");
  scaledCanvas.width = scaledCanvas.height = 20;
  scaledCtx = scaledCanvas.getContext("2d");

  pixelCanvas = document.getElementById("pixelCanvas");
  pixelCtx = pixelCanvas.getContext("2d");

  networkCanvas = document.getElementById("networkCanvas");
  networkCtx = networkCanvas.getContext("2d");

  outputText = document.getElementById("output");

  on_network_ready(() => {
    run_network(false, true);
    drawHereText(drawingCanvas, drawingCtx);
    outputText.innerHTML = "Guess: ?";
  });

  drawingCanvas.addEventListener("pointerdown", function(e) {
    if (drawHereTextShowing) {
      drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
      drawHereTextShowing = false;
    }
    mouseDown = true;
    let rect = drawingCanvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    // use small delta to force canvas to draw a dot
    draw_line(drawingCtx, x - 0.01, y - 0.01, x + 0.01, y + 0.01);
    prevX = x;
    prevY = y;
  });

  drawingCanvas.addEventListener("pointermove", function(e) {
    if (mouseDown) {
      clearCalculatedBoxes();
      run_network(false);
      let rect = drawingCanvas.getBoundingClientRect();
      let x = e.clientX - rect.left;
      let y = e.clientY - rect.top;
      draw_line(drawingCtx, prevX, prevY, x, y);
      prevX = x;
      prevY = y;
    }
    e.preventDefault();
  });

  drawingCanvas.addEventListener("pointerup", function(e) {
    if (mouseDown) {
      mouseDown = false;
      run_network(true);
    }
  });

  drawingCanvas.addEventListener("pointerleave", function(e) {
    mouseDown = false;
  });

  clearBtn.addEventListener("click", function() {
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    clearCalculatedBoxes();
    run_network(false, true);
    outputText.innerHTML = "Guess: ?";
    drawHereText(drawingCanvas, drawingCtx);
  });
});

function drawHereText(canvas, ctx) {
  ctx.font = "30px arial";
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.fillText("draw a digit here", 40, 155);
  drawHereTextShowing = true;
}

function clearCalculatedBoxes() {
  synapseDrawingIterator = undefined;
  scaledCtx.clearRect(0, 0, scaledCanvas.width, scaledCanvas.height);
  networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
  pixelCtx.clearRect(0, 0, pixelCanvas.width, pixelCanvas.height);
}

function draw_line(ctx, x1, y1, x2, y2) {
  ctx.lineWidth = 30;
  ctx.lineJoin = "round";
  ctx.strokeStyle = "black";

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.stroke();
}

function draw_bounding_box(ctx, bounds) {
  ctx.lineWidth = 1;
  ctx.lineJoin = "milter";
  ctx.strokeStyle = "red";
  ctx.beginPath();
  ctx.rect(bounds.left, bounds.top, bounds.width, bounds.height);
  ctx.closePath();
  ctx.stroke();
}

function draw_crosshair(ctx, x, y, radius) {
  ctx.lineJoin = "milter";
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - radius, y);
  ctx.lineTo(x + radius, y);
  ctx.moveTo(x, y - radius);
  ctx.lineTo(x, y + radius);
  ctx.closePath();
  ctx.stroke();
}

function rgba(r, g, b, a) {
  return "rgba(" + Math.floor(r) + "," + Math.floor(g) + "," + Math.floor(b) + "," + a + ")";
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};

function draw_synapse(ctx, x1, y1, x2, y2, magnitude) {
  let alpha = Math.abs(magnitude * 0.3).clamp(0, 1);

  let clamped = (magnitude * 2).clamp(-1, 1);
  ctx.lineWidth = alpha * 3;
  if (clamped < 0) ctx.strokeStyle = rgba(0, 0, 0, alpha);
  else ctx.strokeStyle = rgba(0, 255, 255, alpha);
  
  ctx.lineJoin = "round";

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.stroke();
}

function get_crop_bounds(canvas, ctx) {
  let width = canvas.width;
  let height = canvas.height;
  let stride = 4; // number of color channels
  let data = ctx.getImageData(0, 0, width, height).data;

  let bounds = { left: width, right: 0, top: height, bottom: 0 };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let i = (y * width + x) * stride + 3;
      if (data[i] !== 0) {
        // only checking alpha channel
        bounds.left = Math.min(bounds.left, x);
        bounds.right = Math.max(bounds.right, x);
        bounds.top = Math.min(bounds.top, y);
        bounds.bottom = Math.max(bounds.bottom, y);
      }
    }
  }

  bounds = {
    left: bounds.left,
    top: bounds.top,
    width: bounds.right - bounds.left,
    height: bounds.bottom - bounds.top
  };

  if (bounds.height > bounds.width) {
    bounds.left -= (bounds.height - bounds.width) / 2;
    bounds.width = bounds.height;
  } else {
    bounds.top -= (bounds.width - bounds.height) / 2;
    bounds.height = bounds.width;
  }

  return bounds;
}

function normalize_and_visualize_input() {
  clearCalculatedBoxes();
  let bounds = get_crop_bounds(drawingCanvas, drawingCtx);
  scaledCtx.drawImage(
    drawingCanvas,
    bounds.left,
    bounds.top,
    bounds.width,
    bounds.height,
    0,
    0,
    20,
    20
  );
  let centroid = get_center_of_mass(scaledCanvas, scaledCtx);
  pixelCtx.drawImage(scaledCanvas, 0, 0, 20, 20, 14 - centroid.x, 14 - centroid.y, 20, 20);

  // extract pixel data
  let data = Array.prototype.slice.call(pixelCtx.getImageData(0, 0, 28, 28).data);
  data = data.filter(function(val, idx) {
    return idx % 4 == 3; // Only using alpha channel
  });
  return data.map(function(val, idx) {
    return val / 255.0;
  });
}

function get_center_of_mass(canvas, ctx) {
  let width = canvas.width;
  let height = canvas.height;
  let stride = 4; // number of color channels
  let data = ctx.getImageData(0, 0, width, height).data;
  let sumX = 0;
  let sumY = 0;
  let sumW = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let i = (y * width + x) * stride + 3;
      sumX += data[i] * x;
      sumY += data[i] * y;
      sumW += data[i];
    }
  }
  return { x: sumX / sumW, y: sumY / sumW };
}

function* synapse_drawing_iterator(canvas, ctx, activations, weights, neuron_heights) {
  let startX = 90;
  let neuron_width = 80;
  let neuron_spacing = 240;
  for (let i = 1; i < activations.length; i++) {
    for (let j = 0; j < activations[i - 1].length; j++) {
      let startY = (700 - activations[i - 1].length * neuron_heights[i - 1]) / 2;
      let nextStartY = (700 - activations[i].length * neuron_heights[i]) / 2;
      for (let k = 0; k < activations[i].length; k++) {
        let activation = activations[i - 1][j] * weights[i - 1][k][j];
        if (activation < -0.01 || activation > 0.01) {
          let beginX = startX + (i - 1) * neuron_spacing + neuron_width;
          let beginY = startY + neuron_heights[i - 1] * (j + 0.5);
          let endX = startX + i * neuron_spacing;
          let endY = nextStartY + neuron_heights[i] * (k + 0.5);
          draw_synapse(ctx, beginX, beginY, endX, endY, 2 * activation);
        }
        yield;
      }
    }
  }
}

function draw_network(canvas, ctx, activations, weights, neuron_heights, draw_synapses, reset) {
  let startX = 90;
  let neuron_width = 80;
  let neuron_spacing = 240;
  ctx.strokeStyle = "rgba(0, 0, 0, 0)";
  for (let i = 0; i < activations.length; i++) {
    for (let j = 0; j < activations[i].length; j++) {
      let startY = (700 - activations[i].length * neuron_heights[i]) / 2;

      // Draw neuron activation
      let a = activations[i][j];
      if (reset) {
        // Empty network, fill with black
        ctx.fillStyle = rgba(0, 0, 0, 1.0);
      } else {
        ctx.fillStyle = rgba(0, 255*a, 255*a, 1.0);
      }
      ctx.fillRect(
        startX + i * neuron_spacing,
        startY + j * neuron_heights[i]+0.0001,
        neuron_width,
        neuron_heights[i]
      );
    }
  }
  // Draw labels
  ctx.font = "25px arial";
  ctx.fillStyle = "rgba(0, 0, 0, 1)";

  let startY = (700 - activations[3].length * neuron_heights[3]) / 2;
  for (let i = 0; i < 10; i++) {
    ctx.fillText(i.toString(), 905, startY + neuron_heights[3] / 2 + i * neuron_heights[3] + 8);
  }

  ctx.fillText("pixel", 20, 300);
  ctx.fillText("inputs", 15, 330);
  ctx.fillText("hidden layer 1", 300, 645);
  ctx.fillText("hidden layer 2", 540, 645);
  ctx.fillText("output", 815, 540);
}

function run_network(draw_synapses = false, reset = false) {
  let pixels = normalize_and_visualize_input();
  let activations = evaluate_network(pixels);
  let result = argmax(activations[3]);
  outputText.innerHTML = "Guess: " + result;
  draw_network(
    networkCanvas,
    networkCtx,
    activations,
    network.weights,
    [1, 2, 2, 30],
    draw_synapses,
    reset
  );
  if (draw_synapses) {
    synapseDrawingIterator = synapse_drawing_iterator(networkCanvas, networkCtx, activations, network.weights, [1, 2, 2, 30]);
  }
}

function draw_some_synapses() {
  if (typeof synapseDrawingIterator === 'undefined') {
    return;
  }
  let start = Date.now();
  for (let i = 0; i < 10000; i++) {
    let result = synapseDrawingIterator.next();
    if (result.done) {
      break;
    }
  }
  let end = Date.now();
}

setInterval(draw_some_synapses, 20);
