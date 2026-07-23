struct Boid {
    position: vec2<f32>,
    velocity: vec2<f32>,
};

struct Params {
    dt: f32,
    count: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> source: array<Boid>;
@group(0) @binding(1) var<storage, read_write> destination: array<Boid>;
@group(0) @binding(2) var<uniform> params: Params;

fn limit(vector: vec2<f32>, maximum: f32) -> vec2<f32> {
    let length_squared = dot(vector, vector);
    if length_squared > maximum * maximum {
        return vector * maximum * inverseSqrt(length_squared);
    }
    return vector;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.count { return; }

    let me = source[index];
    var separation = vec2<f32>(0.0);
    var alignment = vec2<f32>(0.0);
    var cohesion = vec2<f32>(0.0);
    var neighbor_count = 0.0;

    for (var other_index = 0u; other_index < params.count; other_index++) {
        if other_index == index { continue; }
        let other = source[other_index];
        let delta = other.position - me.position;
        let distance_squared = dot(delta, delta);
        if distance_squared < 0.015 * 0.015 && distance_squared > 0.000001 {
            separation -= delta / distance_squared;
        }
        if distance_squared < 0.12 * 0.12 {
            alignment += other.velocity;
            cohesion += other.position;
            neighbor_count += 1.0;
        }
    }

    var acceleration = separation * 0.035;
    if neighbor_count > 0.0 {
        acceleration += (alignment / neighbor_count - me.velocity) * 1.4;
        acceleration += (cohesion / neighbor_count - me.position) * 0.8;
    }
    let velocity = limit(me.velocity + acceleration * params.dt, 0.36);
    var position = me.position + velocity * params.dt;
    if position.x < -1.05 { position.x = 1.05; }
    if position.x > 1.05 { position.x = -1.05; }
    if position.y < -1.05 { position.y = 1.05; }
    if position.y > 1.05 { position.y = -1.05; }
    destination[index] = Boid(position, velocity);
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) velocity: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput, @builtin(vertex_index) vertex: u32,
           @builtin(instance_index) instance: u32) -> VertexOutput {
    let direction = normalize(input.velocity);
    let side = vec2<f32>(-direction.y, direction.x);
    var local: vec2<f32>;
    if vertex == 0u { local = direction * 0.030; }
    else if vertex == 1u { local = -direction * 0.018 + side * 0.011; }
    else { local = -direction * 0.018 - side * 0.011; }

    var output: VertexOutput;
    output.position = vec4<f32>(input.position + vec2<f32>(local.x * 0.7, local.y), 0.0, 1.0);
    let variation = f32(instance % 7u) * 0.025;
    output.color = vec4<f32>(0.35, 0.65 + variation, 1.0, 0.9);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
