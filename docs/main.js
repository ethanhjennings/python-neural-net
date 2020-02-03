"use strict";
var mouseDown = false;
var prevX, prevY;

var drawingCanvas;
var drawingCtx;
var boundingCanvas;
var boundingCtx;
var scaledCanvas;
var scaledCtx;
var scaledTLCanvas;
var scaledTLCtx;
var pixelCanvas;
var pixelCtx;
var networkCanvas;
var networkCtx;
var outputText;

var targetTime = 0;

var timeout;

window.addEventListener("DOMContentLoaded", () => {
  drawingCanvas = document.getElementById("drawingCanvas");
  drawingCtx = drawingCanvas.getContext("2d");

  boundingCanvas = document.getElementById("boundingCanvas");
  boundingCtx = boundingCanvas.getContext("2d");

  scaledCanvas = document.getElementById("scaledCanvas");
  scaledCtx = scaledCanvas.getContext("2d");

  scaledTLCanvas = document.getElementById("scaledTopLayerCanvas");
  scaledTLCtx = scaledTLCanvas.getContext("2d");

  pixelCanvas = document.getElementById("pixelCanvas");
  pixelCtx = pixelCanvas.getContext("2d");

  networkCanvas = document.getElementById("networkCanvas");
  networkCtx = networkCanvas.getContext("2d");

  outputText = document.getElementById("output");

  var drawHereTextShowing = true;
  drawHereText(drawingCanvas, drawingCtx);

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
      /*if (typeof timout !== 'undefined') {
              clearTimeout(timeout);
            }*/
      //timeout = setTimeout(function() {run_network();}, 500);
      clearCalculatedBoxes();
      run_network(0.3);
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
      run_network();
    }
  });

  drawingCanvas.addEventListener("pointerleave", function(e) {
    mouseDown = false;
  });

  clearBtn.addEventListener("click", function() {
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    clearCalculatedBoxes();
  });
});

function drawHereText(canvas, ctx) {
  ctx.font = "18px arial";
  ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
  ctx.fillText("Draw a digit here", 30, 100);
}

function clearCalculatedBoxes() {
  outputText.innerHTML = "Guess: ?";
  boundingCtx.clearRect(0, 0, boundingCanvas.width, boundingCanvas.height);
  scaledCtx.clearRect(0, 0, scaledCanvas.width, scaledCanvas.height);
  scaledTLCtx.clearRect(0, 0, scaledTLCanvas.width, scaledTLCanvas.height);
  pixelCtx.clearRect(0, 0, pixelCanvas.width, pixelCanvas.height);
  networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
}

function draw_line(ctx, x1, y1, x2, y2) {
  ctx.lineWidth = 20;
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
  var alpha = Math.abs(magnitude * 0.3).clamp(0, 1);

  var clamped = (magnitude * 2).clamp(-1, 1);
  ctx.lineWidth = alpha * 3;
  if (clamped < 0) ctx.strokeStyle = rgba(0, 0, 0, alpha);
  else ctx.strokeStyle = rgba(0, 255, 255, alpha);

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.stroke();
}

function get_crop_bounds(canvas, ctx) {
  var width = canvas.width;
  var height = canvas.height;
  var stride = 4; // number of color channels
  var data = ctx.getImageData(0, 0, width, height).data;

  var bounds = { left: width, right: 0, top: height, bottom: 0 };
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var i = (y * width + x) * stride + 3;
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
  boundingCtx.drawImage(drawingCanvas, 0, 0);
  draw_bounding_box(boundingCtx, bounds);
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
  draw_crosshair(scaledTLCtx, (centroid.x * 200) / 20, (centroid.y * 200) / 20, 10);

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
  var width = canvas.width;
  var height = canvas.height;
  var stride = 4; // number of color channels
  var data = ctx.getImageData(0, 0, width, height).data;
  var sumX = 0;
  var sumY = 0;
  var sumW = 0;

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var i = (y * width + x) * stride + 3;
      sumX += data[i] * x;
      sumY += data[i] * y;
      sumW += data[i];
    }
  }
  return { x: sumX / sumW, y: sumY / sumW };
}

function draw_network(canvas, ctx, activations, weights, neuron_heights, synapse_epsilon = 0.0) {
  var startX = 90;
  var neuron_width = 80;
  var neuron_spacing = 240;
  for (var i = 0; i < activations.length; i++) {
    for (var j = 0; j < activations[i].length; j++) {
      var startY = (600 - activations[i].length * neuron_heights[i]) / 2;

      // Draw neuron activation
      var a = activations[i][j];
      ctx.fillStyle = rgba(0, 255 * a, 255 * a, 1);
      ctx.fillRect(
        startX + i * neuron_spacing,
        startY + j * neuron_heights[i],
        neuron_width,
        neuron_heights[i]
      );

      // Draw input synapses
      if (i > 0) {
        ctx.lineJoin = "round";
        var prevStartY = (600 - activations[i - 1].length * neuron_heights[i - 1]) / 2;
        for (var k = 0; k < activations[i - 1].length; k++) {
          var activation = activations[i - 1][k] * weights[i - 1][j][k];

          var beginX = startX + (i - 1) * neuron_spacing + neuron_width;
          var beginY = prevStartY + neuron_heights[i - 1] * (k + 0.5);
          var endX = startX + i * neuron_spacing;
          var endY = startY + neuron_heights[i] * (j + 0.5);
          if (activation > synapse_epsilon || activation < -synapse_epsilon) {
            draw_synapse(ctx, beginX, beginY, endX, endY, 2 * activation);
          }
        }
      }
    }
  }
  // Draw labels
  ctx.font = "25px Arial";
  ctx.fillStyle = "rgba(0, 0, 0, 1)";

  var startY = (600 - activations[3].length * neuron_heights[3]) / 2;
  for (i = 0; i < 10; i++) {
    ctx.fillText(i.toString(), 905, startY + neuron_heights[3] / 2 + i * neuron_heights[3] + 8);
  }

  ctx.fillText("pixel", 20, 300);
  ctx.fillText("inputs", 15, 330);
  ctx.fillText("hidden layer 1", 300, 560);
  ctx.fillText("hidden layer 2", 540, 560);
  ctx.fillText("output", 815, 482);
}

function run_network(synapse_epsilon = 0) {
  var pixels = normalize_and_visualize_input();
  var activations = evaluate_network(pixels);
  var result = argmax(activations[3]);
  outputText.innerHTML = "Guess: " + result;
  draw_network(
    networkCanvas,
    networkCtx,
    activations,
    network.weights,
    [0.7, 1.8, 1.8, 30],
    synapse_epsilon
  );
}
