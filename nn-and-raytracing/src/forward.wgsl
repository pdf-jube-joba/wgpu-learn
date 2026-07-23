struct Constants {
    seed: u32,
    pixel_len: u32,
    sample_ray: u32,
    middle_num: u32,
    batch_size: u32,
    rate: f32,
}

@group(0) @binding(0)
var<uniform> constants: Constants;

@group(1) @binding(0)
var<storage, read> images: array<f32>;

// [input -> hidden weights][hidden biases]
@group(1) @binding(1)
var<storage, read> weights1: array<f32>;

@group(1) @binding(2)
var<storage, read_write> middle: array<f32>;

// [hidden -> output weights][output bias]
@group(1) @binding(3)
var<storage, read> weights2: array<f32>;

@group(1) @binding(4)
var<storage, read_write> predicts: array<f32>;

fn input_count() -> u32 {
    return constants.pixel_len * constants.pixel_len;
}

@compute @workgroup_size(8, 8, 1)
fn forward_middle(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let hidden_index = id.x;
    let sample_index = id.y;
    if (
        hidden_index >= constants.middle_num
        || sample_index >= constants.batch_size
    ) {
        return;
    }

    let inputs = input_count();
    let image_offset = sample_index * inputs;
    let weight_offset = hidden_index * inputs;
    let bias_offset = constants.middle_num * inputs;
    var sum = weights1[bias_offset + hidden_index];
    for (var input_index = 0u; input_index < inputs; input_index += 1u) {
        sum += images[image_offset + input_index]
            * weights1[weight_offset + input_index];
    }

    middle[sample_index * constants.middle_num + hidden_index] = tanh(sum);
}

@compute @workgroup_size(64, 1, 1)
fn forward_output(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let sample_index = id.x;
    if (sample_index >= constants.batch_size) {
        return;
    }

    let middle_offset = sample_index * constants.middle_num;
    var sum = weights2[constants.middle_num];
    for (
        var hidden_index = 0u;
        hidden_index < constants.middle_num;
        hidden_index += 1u
    ) {
        sum += middle[middle_offset + hidden_index] * weights2[hidden_index];
    }
    predicts[sample_index] = sum;
}
