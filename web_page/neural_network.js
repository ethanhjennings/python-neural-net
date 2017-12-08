var network = null;

$.getJSON( "network.json", function( data ) {
    network = data;
});

function dot(matrix, vec) {
    outputVec = [];
    for (var i = 0; i < matrix.length; i++) {
	var sum = 0;
        for (var c = 0; c < vec.length; c++) {
	    sum += matrix[i][c]*vec[c];
        }
	outputVec.push(sum);
    }
    return outputVec
}

function add(vec1, vec2) {
    for (var i = 0; i < vec1.length; i++) {
	vec1[i] += vec2[i];
    }
    return vec1;
}

function sigmoid(vec) {
    for (var i = 0; i < vec.length; i++) {
    	vec[i] = 1.0 / (1.0 + Math.exp(-1.0 * vec[i]));
    }
    return vec;
}

function argmax(vec) {
   return vec.indexOf(vec.reduce(function (a, b) { return Math.max(a,b) }));
}

function feed_forward(inputs) {
    activations = [inputs];
    weighted_inputs = [];
    for (var i = 0; i < network.weights.length; i++) {
	var weights = network.weights[i];
	var biases = network.biases[i];
	var layer_input = add(dot(weights, activations[i]), biases);
        var layer_activations = sigmoid(layer_input);
	activations.push(layer_activations);
	weighted_inputs.push(layer_input);
    }
    return {activations: activations, weighted_inputs: weighted_inputs};
}

function evaluate_network(inputs) {
    return feed_forward(inputs);
}
