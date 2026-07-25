struct Constants {
    pixel_len: u32,
    hidden_size: u32,
    batch_size: u32,
}

@group(0) @binding(0)
var<uniform> constants: Constants;

@group(1) @binding(0)
var<storage, read> images: array<f32>;

// [input -> hidden weights][hidden biases]
@group(1) @binding(1)
var<storage, read> weights1: array<f32>;

@group(1) @binding(2)
var<storage, read_write> hidden: array<f32>;

// [hidden -> output weights][output bias]
@group(1) @binding(3)
var<storage, read> weights2: array<f32>;

@group(1) @binding(4)
var<storage, read_write> predicts: array<f32>;

fn input_count() -> u32 {
    return constants.pixel_len * constants.pixel_len;
}

@compute @workgroup_size(8, 8, 1)
fn forward_hidden(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let hidden_index = id.x;
    let sample_index = id.y;
    if (
        hidden_index >= constants.hidden_size
        || sample_index >= constants.batch_size
    ) {
        return;
    }

    let inputs = input_count();
    let image_offset = sample_index * inputs;
    let weight_offset = hidden_index * inputs;
    let bias_offset = constants.hidden_size * inputs;
    var sum = weights1[bias_offset + hidden_index];
    for (var input_index = 0u; input_index < inputs; input_index += 1u) {
        sum += images[image_offset + input_index]
            * weights1[weight_offset + input_index];
    }

    hidden[sample_index * constants.hidden_size + hidden_index] = tanh(sum);
}

@compute @workgroup_size(64, 1, 1)
fn forward_output(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let sample_index = id.x;
    if (sample_index >= constants.batch_size) {
        return;
    }

    let hidden_offset = sample_index * constants.hidden_size;
    var sum = weights2[constants.hidden_size];
    for (
        var hidden_index = 0u;
        hidden_index < constants.hidden_size;
        hidden_index += 1u
    ) {
        sum += hidden[hidden_offset + hidden_index] * weights2[hidden_index];
    }
    predicts[sample_index] = sum;
}
