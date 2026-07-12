struct Grid {
    values: array<f32>,
}

struct Params {
    width: u32,
    height: u32,
    alpha: f32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> src: Grid;

@group(0) @binding(1)
var<storage, read_write> dst: Grid;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let row_start = y * params.width;
    let index = row_start + x;
    let center = src.values[index];

    if (x == 0u || y == 0u || x + 1u == params.width || y + 1u == params.height) {
        dst.values[index] = center;
        return;
    }

    let left = src.values[index - 1u];
    let right = src.values[index + 1u];
    let up = src.values[index - params.width];
    let down = src.values[index + params.width];
    let laplacian = left + right + up + down - 4.0 * center;

    dst.values[index] = center + params.alpha * laplacian;
}
