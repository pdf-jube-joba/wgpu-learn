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
var<storage, read_write> samples: array<f32>;

@group(1) @binding(1)
var<storage, read_write> expects: array<f32>;

@group(1) @binding(2)
var<storage, read_write> images: array<f32>;

@group(1) @binding(3)
var debug_images: texture_storage_2d_array<rgba8unorm, write>;

fn next_u32(seed: ptr<function, u32>) -> u32 {
    (*seed) = (*seed) * 1664525u + 1013904223u;
    return *seed;
}

fn random_f32(seed: ptr<function, u32>) -> f32 {
    // Use the upper 24 bits so every result can be represented exactly as f32.
    return f32(next_u32(seed) >> 8u) * (1.0 / 16777216.0);
}

fn mix_u32(value: u32) -> u32 {
    var mixed = value;
    mixed ^= mixed >> 16u;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15u;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16u;
    return mixed;
}

const SAMPLE_STRIDE: u32 = 3u;
const SPHERE_RADIUS: f32 = 1.0;
const MIN_DISTANCE: f32 = 1.0;
const MAX_DISTANCE: f32 = 10.0;
const MAX_ANGLE: f32 = 0.95;
const SCREEN_DISTANCE: f32 = 1.0;
const SCREEN_HALF_EXTENT: f32 = 1.0;
const MIN_VISIBLE_FRACTION: f32 = 0.30;
const MAX_IMAGE_FRACTION: f32 = 0.30;
const MAX_GENERATE_RETRIES: u32 = 64u;
const PI: f32 = 3.141592653589793;
const FALLBACK_CENTER: vec3<f32> = vec3<f32>(0.0, 6.0, 0.0);

struct ProjectedEllipse {
    center: vec2<f32>,
    parallel_axis: vec2<f32>,
    parallel_radius: f32,
    perpendicular_radius: f32,
}

fn project_sphere(center: vec3<f32>) -> ProjectedEllipse {
    let lateral = vec2<f32>(center.x, center.z);
    let lateral_length = length(lateral);
    var parallel_axis = vec2<f32>(1.0, 0.0);
    if (lateral_length > 0.000001) {
        parallel_axis = lateral / lateral_length;
    }

    let radius_squared = SPHERE_RADIUS * SPHERE_RADIUS;
    let q = dot(center, center) - radius_squared;
    let depth_term = center.y * center.y - radius_squared;

    return ProjectedEllipse(
        SCREEN_DISTANCE * center.y * lateral / depth_term,
        parallel_axis,
        SCREEN_DISTANCE * SPHERE_RADIUS * sqrt(q) / depth_term,
        SCREEN_DISTANCE * SPHERE_RADIUS / sqrt(depth_term),
    );
}

fn cross_2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn circle_sector_area(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return 0.5 * atan2(cross_2d(a, b), dot(a, b));
}

// Signed area contributed by one polygon edge intersected with the unit disk.
// The edge is split at its circle intersections. A sub-edge inside the circle
// contributes a triangle; one outside contributes a circular sector.
fn unit_circle_edge_area(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let direction = b - a;
    let quadratic_a = dot(direction, direction);
    if (quadratic_a <= 0.0000001) {
        return 0.0;
    }

    var split: array<f32, 4>;
    var split_count = 1u;
    split[0] = 0.0;

    let quadratic_b = 2.0 * dot(a, direction);
    let quadratic_c = dot(a, a) - 1.0;
    let discriminant = quadratic_b * quadratic_b
        - 4.0 * quadratic_a * quadratic_c;

    if (discriminant > 0.0) {
        let root = sqrt(discriminant);
        let first = (-quadratic_b - root) / (2.0 * quadratic_a);
        let second = (-quadratic_b + root) / (2.0 * quadratic_a);
        if (first > 0.0 && first < 1.0) {
            split[split_count] = first;
            split_count += 1u;
        }
        if (second > 0.0 && second < 1.0) {
            split[split_count] = second;
            split_count += 1u;
        }
    }

    split[split_count] = 1.0;
    split_count += 1u;

    var area = 0.0;
    for (var index = 0u; index + 1u < split_count; index += 1u) {
        let start = a + direction * split[index];
        let end = a + direction * split[index + 1u];
        let middle = (start + end) * 0.5;
        if (dot(middle, middle) <= 1.0) {
            area += 0.5 * cross_2d(start, end);
        } else {
            area += circle_sector_area(start, end);
        }
    }
    return area;
}

fn ellipse_local_point(
    point: vec2<f32>,
    ellipse: ProjectedEllipse,
) -> vec2<f32> {
    let perpendicular_axis = vec2<f32>(
        -ellipse.parallel_axis.y,
        ellipse.parallel_axis.x,
    );
    let relative = point - ellipse.center;
    return vec2<f32>(
        dot(relative, ellipse.parallel_axis) / ellipse.parallel_radius,
        dot(relative, perpendicular_axis) / ellipse.perpendicular_radius,
    );
}

fn visible_ellipse_area(ellipse: ProjectedEllipse) -> f32 {
    // Transforming the ellipse to a unit circle turns the screen rectangle into
    // this four-sided polygon. Its winding remains counter-clockwise.
    var corners: array<vec2<f32>, 4>;
    corners[0] = ellipse_local_point(
        vec2<f32>(-SCREEN_HALF_EXTENT, -SCREEN_HALF_EXTENT), ellipse,
    );
    corners[1] = ellipse_local_point(
        vec2<f32>( SCREEN_HALF_EXTENT, -SCREEN_HALF_EXTENT), ellipse,
    );
    corners[2] = ellipse_local_point(
        vec2<f32>( SCREEN_HALF_EXTENT,  SCREEN_HALF_EXTENT), ellipse,
    );
    corners[3] = ellipse_local_point(
        vec2<f32>(-SCREEN_HALF_EXTENT,  SCREEN_HALF_EXTENT), ellipse,
    );

    var unit_circle_area = 0.0;
    for (var edge = 0u; edge < 4u; edge += 1u) {
        unit_circle_area += unit_circle_edge_area(
            corners[edge],
            corners[(edge + 1u) % 4u],
        );
    }

    return abs(unit_circle_area)
        * ellipse.parallel_radius
        * ellipse.perpendicular_radius;
}

fn acceptable_sample(center: vec3<f32>, distance: f32) -> bool {
    // Projection is an ellipse only while the complete sphere is in front of
    // the camera. The generated range normally guarantees this; keep the check
    // here so later constant changes cannot produce invalid square roots.
    if (distance <= SPHERE_RADIUS || center.y <= SPHERE_RADIUS) {
        return false;
    }

    let ellipse = project_sphere(center);
    let projected_area = PI
        * ellipse.parallel_radius
        * ellipse.perpendicular_radius;
    let visible_area = visible_ellipse_area(ellipse);
    let screen_area = 4.0 * SCREEN_HALF_EXTENT * SCREEN_HALF_EXTENT;
    let image_fraction = visible_area / screen_area;
    let visible_fraction = visible_area / projected_area;

    return image_fraction <= MAX_IMAGE_FRACTION
        && visible_fraction >= MIN_VISIBLE_FRACTION;
}

fn camera_ray_direction(pixel: vec2<f32>) -> vec3<f32> {
    let image_size = f32(constants.pixel_len);
    let screen_x = (
        pixel.x / image_size * 2.0 - 1.0
    ) * SCREEN_HALF_EXTENT;
    let screen_z = (
        1.0 - pixel.y / image_size * 2.0
    ) * SCREEN_HALF_EXTENT;
    return normalize(vec3<f32>(screen_x, SCREEN_DISTANCE, screen_z));
}

fn sphere_intersection(
    ray_direction: vec3<f32>,
    sphere_center: vec3<f32>,
) -> f32 {
    // Camera/ray origin is (0, 0, 0).
    let camera_to_center = -sphere_center;
    let b = dot(camera_to_center, ray_direction);
    let c = dot(camera_to_center, camera_to_center)
        - SPHERE_RADIUS * SPHERE_RADIUS;
    let discriminant = b * b - c;
    if (discriminant < 0.0) {
        return -1.0;
    }

    let root = sqrt(discriminant);
    let near = -b - root;
    if (near >= 0.0) {
        return near;
    }

    let far = -b + root;
    if (far >= 0.0) {
        return far;
    }
    return -1.0;
}

fn trace_ray(
    ray_direction: vec3<f32>,
    sphere_center: vec3<f32>,
) -> f32 {
    let hit_distance = sphere_intersection(ray_direction, sphere_center);
    if (hit_distance < 0.0) {
        return 0.0;
    }

    let hit_position = ray_direction * hit_distance;
    let surface_normal = normalize(hit_position - sphere_center);

    // The point light is at the camera. Its intensity does not attenuate with
    // distance, as specified by the example.
    let light_direction = normalize(-hit_position);
    return clamp(dot(surface_normal, light_direction), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn generate_samples(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let sample_index = id.x;
    if (sample_index >= constants.batch_size) {
        return;
    }

    // Mixing the invocation index first avoids strongly correlated streams for
    // adjacent samples while keeping each invocation independent.
    var seed = mix_u32(constants.seed ^ sample_index);

    var center = FALLBACK_CENTER;
    var distance = length(FALLBACK_CENTER);

    for (var retry = 0u; retry < MAX_GENERATE_RETRIES; retry += 1u) {
        let candidate_distance = mix(
            MIN_DISTANCE,
            MAX_DISTANCE,
            random_f32(&seed),
        );
        let horizontal_angle = mix(
            -MAX_ANGLE,
            MAX_ANGLE,
            random_f32(&seed),
        );
        let vertical_angle = mix(
            -MAX_ANGLE,
            MAX_ANGLE,
            random_f32(&seed),
        );

        // Left-handed coordinates: +Y is forward and +Z is up.
        let cos_vertical = cos(vertical_angle);
        let direction = vec3<f32>(
            sin(horizontal_angle) * cos_vertical,
            cos(horizontal_angle) * cos_vertical,
            sin(vertical_angle),
        );
        let candidate_center = direction * candidate_distance;

        if (acceptable_sample(candidate_center, candidate_distance)) {
            center = candidate_center;
            distance = candidate_distance;
            break;
        }
    }

    // Each invocation owns exactly this three-f32 slice of samples.
    let sample_offset = sample_index * SAMPLE_STRIDE;
    samples[sample_offset + 0u] = center.x;
    samples[sample_offset + 1u] = center.y;
    samples[sample_offset + 2u] = center.z;

    expects[sample_index] = distance;
}

@compute @workgroup_size(8, 8, 1)
fn raytrace(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let pixel_x = id.x;
    let pixel_y = id.y;
    let sample_index = id.z;
    if (
        pixel_x >= constants.pixel_len
        || pixel_y >= constants.pixel_len
        || sample_index >= constants.batch_size
    ) {
        return;
    }

    let sample_offset = sample_index * SAMPLE_STRIDE;
    let sphere_center = vec3<f32>(
        samples[sample_offset + 0u],
        samples[sample_offset + 1u],
        samples[sample_offset + 2u],
    );

    let ray_count = max(constants.sample_ray, 1u);
    let pixel_index = pixel_y * constants.pixel_len + pixel_x;
    var seed = mix_u32(
        constants.seed
        ^ mix_u32(sample_index)
        ^ mix_u32(pixel_index),
    );
    var sum = 0.0;

    for (var ray_index = 0u; ray_index < ray_count; ray_index += 1u) {
        var offset = vec2<f32>(0.5, 0.5);
        if (ray_count > 1u) {
            offset = vec2<f32>(random_f32(&seed), random_f32(&seed));
        }

        let pixel = vec2<f32>(f32(pixel_x), f32(pixel_y)) + offset;
        let ray_direction = camera_ray_direction(pixel);
        sum += trace_ray(ray_direction, sphere_center);
    }

    let image_offset = sample_index
        * constants.pixel_len
        * constants.pixel_len;
    let color = sum / f32(ray_count);
    images[image_offset + pixel_index] = color;
    textureStore(
        debug_images,
        vec2<i32>(i32(pixel_x), i32(pixel_y)),
        i32(sample_index),
        vec4<f32>(color, color, color, 1.0),
    );
}
