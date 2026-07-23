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

@group(1) @binding(1)
var<storage, read> expects: array<f32>;

@group(1) @binding(2)
var<storage, read> middle: array<f32>;

@group(1) @binding(3)
var<storage, read> predicts: array<f32>;

// [input -> hidden weights][hidden biases]
@group(1) @binding(4)
var<storage, read_write> weights1: array<f32>;

// [hidden -> output weights][output bias]
@group(1) @binding(5)
var<storage, read_write> weights2: array<f32>;

@group(1) @binding(6)
var<storage, read_write> hidden_delta: array<f32>;

@group(1) @binding(7)
var<storage, read_write> loss: array<f32>;

fn input_count() -> u32 {
    return constants.pixel_len * constants.pixel_len;
}

fn output_delta(sample_index: u32) -> f32 {
    return 2.0 * (predicts[sample_index] - expects[sample_index])
        / f32(constants.batch_size);
}

@compute @workgroup_size(8, 8, 1)
fn backward_hidden(
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

    let index = sample_index * constants.middle_num + hidden_index;
    let activation = middle[index];
    hidden_delta[index] = output_delta(sample_index)
        * weights2[hidden_index]
        * (1.0 - activation * activation);
}

@compute @workgroup_size(64, 1, 1)
fn update_weights(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let parameter_index = id.x;
    let inputs = input_count();
    let layer1_weight_count = constants.middle_num * inputs;
    let layer1_parameter_count = layer1_weight_count + constants.middle_num;
    let layer2_parameter_count = constants.middle_num + 1u;
    if (parameter_index >= layer1_parameter_count + layer2_parameter_count) {
        return;
    }

    var gradient = 0.0;
    if (parameter_index < layer1_weight_count) {
        let hidden_index = parameter_index / inputs;
        let input_index = parameter_index % inputs;
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += hidden_delta[
                sample_index * constants.middle_num + hidden_index
            ] * images[sample_index * inputs + input_index];
        }
        weights1[parameter_index] -= constants.rate * gradient;
        return;
    }

    if (parameter_index < layer1_parameter_count) {
        let hidden_index = parameter_index - layer1_weight_count;
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += hidden_delta[
                sample_index * constants.middle_num + hidden_index
            ];
        }
        weights1[parameter_index] -= constants.rate * gradient;
        return;
    }

    let layer2_index = parameter_index - layer1_parameter_count;
    if (layer2_index < constants.middle_num) {
        for (
            var sample_index = 0u;
            sample_index < constants.batch_size;
            sample_index += 1u
        ) {
            gradient += output_delta(sample_index)
                * middle[sample_index * constants.middle_num + layer2_index];
        }
        weights2[layer2_index] -= constants.rate * gradient;
        return;
    }

    for (
        var sample_index = 0u;
        sample_index < constants.batch_size;
        sample_index += 1u
    ) {
        gradient += output_delta(sample_index);
    }
    weights2[constants.middle_num] -= constants.rate * gradient;
}

@compute @workgroup_size(1, 1, 1)
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
