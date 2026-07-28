struct Constants {
    input_size: u32,
    hidden1_size: u32,
    hidden2_size: u32,
    batch_size: u32,
}

@group(0) @binding(0)
var<uniform> constants: Constants;

@group(1) @binding(0)
var<storage, read> images: array<f32>;

// [input -> hidden1 weights][hidden1 biases]
@group(1) @binding(1)
var<storage, read> weights1: array<f32>;

@group(1) @binding(2)
var<storage, read_write> hidden1: array<f32>;

// [hidden1 -> hidden2 weights][hidden2 biases]
@group(1) @binding(3)
var<storage, read> weights2: array<f32>;

@group(1) @binding(4)
var<storage, read_write> hidden2: array<f32>;

// [hidden2 -> output weights][output bias]
@group(1) @binding(5)
var<storage, read> weights3: array<f32>;

@group(1) @binding(6)
var<storage, read_write> predicts: array<f32>;

@compute @workgroup_size(8, 8)
fn forward_hidden1(
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

    let image_offset = sample_index * constants.input_size;
    let weight_offset = hidden_index * constants.input_size;
    let bias_offset = constants.hidden1_size * constants.input_size;
    var sum = weights1[bias_offset + hidden_index];
    for (
        var input_index = 0u;
        input_index < constants.input_size;
        input_index += 1u
    ) {
        sum += images[image_offset + input_index]
            * weights1[weight_offset + input_index];
    }

    hidden1[sample_index * constants.hidden1_size + hidden_index] = tanh(sum);
}

@compute @workgroup_size(8, 8)
fn forward_hidden2(
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

    let hidden1_offset = sample_index * constants.hidden1_size;
    let weight_offset = hidden_index * constants.hidden1_size;
    let bias_offset = constants.hidden2_size * constants.hidden1_size;
    var sum = weights2[bias_offset + hidden_index];
    for (
        var source_index = 0u;
        source_index < constants.hidden1_size;
        source_index += 1u
    ) {
        sum += hidden1[hidden1_offset + source_index]
            * weights2[weight_offset + source_index];
    }

    hidden2[sample_index * constants.hidden2_size + hidden_index] = tanh(sum);
}

@compute @workgroup_size(64)
fn forward_output(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let sample_index = id.x;
    if (sample_index >= constants.batch_size) {
        return;
    }

    let hidden_offset = sample_index * constants.hidden2_size;
    var sum = weights3[constants.hidden2_size];
    for (
        var hidden_index = 0u;
        hidden_index < constants.hidden2_size;
        hidden_index += 1u
    ) {
        sum += hidden2[hidden_offset + hidden_index] * weights3[hidden_index];
    }
    predicts[sample_index] = sum;
}
