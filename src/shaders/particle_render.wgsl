@group(0) @binding(0) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(1) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(2) var<uniform> modelMatrix: mat4x4<f32>;

struct VertexInput {
  @location(0) pos: vec4<f32>, // mesh vertex position (billboard quad) : slot 0
  @location(3) instancePos: vec4<f32>, // per-particle position (xyz), w unused : slot 1
  @location(4) instanceRot: vec4<f32>, // per-particle rotation quaternion : slot 2
  @location(5) instanceVel: vec4<f32>, // per-particle velocity (xyz), w unused : slot 3
};

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(1) localPos: vec4<f32>,
};

// Quaternion rotation helper
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let u = q.xyz;
  let s = q.w;
  return 2.0 * dot(u, v) * u
       + (s * s - dot(u, u)) * v
       + 2.0 * s * cross(u, v);
}

@vertex
fn main_vertex(input: VertexInput) -> VertexOutput {
  var velocityMag = length(input.instanceVel.xyz);

  velocityMag = mix(0.01, 1.0, velocityMag);
  // use power to make velocityMag more extreme
  velocityMag = pow(velocityMag, 2.0);

  let scale = 0.04;

  let scaledPos = input.pos * vec4<f32>(velocityMag * 3.0, 0.01, 0.01, 1.0);
  let rotated = quat_rotate(input.instanceRot, scaledPos.xyz);
  // let rotated = scaledPos.xyz;
  let localWorld = rotated + input.instancePos.xyz;
  let worldPos = modelMatrix * vec4<f32>(localWorld, 1.0);
  return VertexOutput(
    projectionMatrix * viewMatrix * worldPos,
    input.pos,
  );
}

@fragment
fn main_fragment(
  @location(1) localPos: vec4<f32>
) -> @location(0) vec4<f32> {
  // Use the normalized local mesh position as a 'normal' for color
  let normal = normalize(localPos.xyz);
  let color = normal * 0.5 + 0.5;
  return vec4<f32>(color, 1.0);
}