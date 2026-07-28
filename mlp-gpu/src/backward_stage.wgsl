struct Constants {
    input_size: u32,
    hidden1_size: u32,
    hidden2_size: u32,
    batch_size: u32,
    rate: f32,
}

@group(0) @binding(0)
var<uniform> constants: Constants;

@group(1) @binding(0)
var<storage, read> images: array<f32>;

@group(1) @binding(1)
var<storage, read> expects: array<f32>;

@group(1) @binding(2)
var<storage, read> hidden1: array<f32>;

@group(1) @binding(3)
var<storage, read> predicts: array<f32>;

// [input -> hidden1 weights][hidden1 biases]
@group(1) @binding(4)
var<storage, read_write> weights1: array<f32>;

// [hidden1 -> hidden2 weights][hidden2 biases]
@group(1) @binding(5)
var<storage, read_write> weights2: array<f32>;

@group(1) @binding(6)
var<storage, read_write> hidden1_delta: array<f32>;

@group(1) @binding(7)
var<storage, read_write> loss: array<f32>;

@group(1) @binding(8)
var<storage, read> hidden2: array<f32>;

// [hidden2 -> output weights][output bias]
@group(1) @binding(9)
var<storage, read_write> weights3: array<f32>;

@group(1) @binding(10)
var<storage, read_write> hidden2_delta: array<f32>;

fn output_delta(sample_index: u32) -> f32 {
    return 2.0 * (predicts[sample_index] - expects[sample_index])
        / f32(constants.batch_size);
}

@compute @workgroup_size(8, 8)
fn backward_hidden2(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let hidden_index = id.x;
    let sample_index = id.y;
    if (
        hidden_index >= constants.hidden2_size
        || sample_index >= constants.batch_size
    ) {
        return;
    }

    let index = sample_index * constants.hidden2_size + hidden_index;
    let activation = hidden2[index];
    hidden2_delta[index] = output_delta(sample_index)
        * weights3[hidden_index]
        * (1.0 - activation * activation);
}

@compute @workgroup_size(8, 8)
fn backward_hidden1(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let hidden_index = id.x;
    let sample_index = id.y;
    if (
        hidden_index >= constants.hidden1_size
        || sample_index >= constants.batch_size
    ) {
        return;
    }

    var downstream = 0.0;
    for (
        var hidden2_index = 0u;
        hidden2_index < constants.hidden2_size;
        hidden2_index += 1u
    ) {
        downstream += hidden2_delta[
            sample_index * constants.hidden2_size + hidden2_index
        ] * weights2[hidden2_index * constants.hidden1_size + hidden_index];
    }

    let index = sample_index * constants.hidden1_size + hidden_index;
    let activation = hidden1[index];
    hidden1_delta[index] = downstream * (1.0 - activation * activation);
}

@compute @workgroup_size(64)
fn update_weights1(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let parameter_index = id.x;
    let layer1_weight_count = constants.hidden1_size * constants.input_size;
    let layer1_parameter_count = layer1_weight_count + constants.hidden1_size;
    if (parameter_index >= layer1_parameter_count) {
        return;
    }

    var gradient = 0.0;
    if (parameter_index < layer1_weight_count) {
        let hidden_index = parameter_index / constants.input_size;
        let input_index = parameter_index % constants.input_size;
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += hidden1_delta[
                sample_index * constants.hidden1_size + hidden_index
            ] * images[sample_index * constants.input_size + input_index];
        }
        weights1[parameter_index] -= constants.rate * gradient;
        return;
    }

    let hidden_index = parameter_index - layer1_weight_count;
    for (
        var sample_index = 0u;
        sample_index < constants.batch_size;
        sample_index += 1u
    ) {
        gradient += hidden1_delta[
            sample_index * constants.hidden1_size + hidden_index
        ];
    }
    weights1[parameter_index] -= constants.rate * gradient;
}

@compute @workgroup_size(64)
fn update_weights2(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let parameter_index = id.x;
    let layer2_weight_count = constants.hidden2_size * constants.hidden1_size;
    let layer2_parameter_count = layer2_weight_count + constants.hidden2_size;
    if (parameter_index >= layer2_parameter_count) {
        return;
    }

    var gradient = 0.0;
    if (parameter_index < layer2_weight_count) {
        let hidden2_index = parameter_index / constants.hidden1_size;
        let hidden1_index = parameter_index % constants.hidden1_size;
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += hidden2_delta[
                sample_index * constants.hidden2_size + hidden2_index
            ] * hidden1[sample_index * constants.hidden1_size + hidden1_index];
        }
        weights2[parameter_index] -= constants.rate * gradient;
        return;
    }

    let hidden2_index = parameter_index - layer2_weight_count;
    for (
        var sample_index = 0u;
        sample_index < constants.batch_size;
        sample_index += 1u
    ) {
        gradient += hidden2_delta[
            sample_index * constants.hidden2_size + hidden2_index
        ];
    }
    weights2[parameter_index] -= constants.rate * gradient;
}

@compute @workgroup_size(64)
fn update_weights3(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let parameter_index = id.x;
    if (parameter_index > constants.hidden2_size) {
        return;
    }

    var gradient = 0.0;
    if (parameter_index < constants.hidden2_size) {
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += output_delta(sample_index)
                * hidden2[sample_index * constants.hidden2_size + parameter_index];
        }
        weights3[parameter_index] -= constants.rate * gradient;
        return;
    }

    for (
        var sample_index = 0u;
        sample_index < constants.batch_size;
        sample_index += 1u
    ) {
        gradient += output_delta(sample_index);
    }
    weights3[constants.hidden2_size] -= constants.rate * gradient;
}

@compute @workgroup_size(1)
fn compute_loss() {
    var sum = 0.0;
    for (
        var sample_index = 0u;
        sample_index < constants.batch_size;
        sample_index += 1u
    ) {
        let difference = predicts[sample_index] - expects[sample_index];
        sum += difference * difference;
    }
    loss[0] = sum / f32(constants.batch_size);
}
