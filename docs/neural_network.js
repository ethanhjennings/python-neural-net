var network = null;
var ready_callback = null;

function get_json(url, callback) {
  let xhr = new XMLHttpRequest();
  xhr.open('GET', url, true);
  xhr.onreadystatechange = function() {
    if(xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
      let json = JSON.parse(xhr.responseText);
      callback(json);
      ready_callback();
    }
  }
  xhr.send();
}

get_json("network.json", function(data) {
  network = data;
});

function on_network_ready(callback) {
  ready_callback = callback;
  if (network !== null) {
    // Already done loading!
    ready_callback();
  }
}

function dot(matrix, vec) {
  outputVec = [];
  for (var i = 0; i < matrix.length; i++) {
    var sum = 0;
    for (var c = 0; c < vec.length; c++) {
      sum += matrix[i][c] * vec[c];
    }
    outputVec.push(sum);
  }
  return outputVec;
}

function add(vec1, vec2) {
  for (var i = 0; i < vec1.length; i++) {
    vec1[i] += vec2[i];
  }
  return vec1;
}

function relu(vec) {
  for (var i = 0; i < vec.length; i++) {
    vec[i] = Math.max(vec[i], 0);
  }
  return vec;
}

function argmax(vec) {
  return vec.indexOf(
    vec.reduce(function(a, b) {
      return Math.max(a, b);
    })
  );
}

function feed_forward(inputs) {
  activations = [inputs];
  for (var i = 0; i < network.weights.length; i++) {
    var weights = network.weights[i];
    var biases = network.biases[i];
    var layer_input = add(dot(weights, activations[i]), biases);
    var layer_activations = relu(layer_input);
    activations.push(layer_activations);
  }
  return activations;
}

function evaluate_network(inputs) {
  return feed_forward(inputs);
}
