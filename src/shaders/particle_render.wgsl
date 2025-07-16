@group(0) @binding(0) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(1) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(2) var<uniform> modelMatrix: mat4x4<f32>;

struct VertexInput {
  @location(0) pos: vec4<f32>, // mesh vertex position (billboard quad) : slot 0
  @location(3) instancePos: vec4<f32>, // per-particle position (xyz), w unused : slot 1
  @location(4) instanceVel: vec4<f32>, // per-particle velocity (xyz), w unused : slot 2
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

// Convert a 3x3 rotation matrix to a quaternion
fn quat_from_matrix(m: mat3x3<f32>) -> vec4<f32> {
  let trace = m[0][0] + m[1][1] + m[2][2];
  var q = vec4<f32>(0.0);
  if (trace > 0.0) {
    let s = sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (m[2][1] - m[1][2]) / s;
    q.y = (m[0][2] - m[2][0]) / s;
    q.z = (m[1][0] - m[0][1]) / s;
  } else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2])) {
    let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
    q.w = (m[2][1] - m[1][2]) / s;
    q.x = 0.25 * s;
    q.y = (m[0][1] + m[1][0]) / s;
    q.z = (m[0][2] + m[2][0]) / s;
  } else if (m[1][1] > m[2][2]) {
    let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
    q.w = (m[0][2] - m[2][0]) / s;
    q.x = (m[0][1] + m[1][0]) / s;
    q.y = 0.25 * s;
    q.z = (m[1][2] + m[2][1]) / s;
  } else {
    let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
    q.w = (m[1][0] - m[0][1]) / s;
    q.x = (m[0][2] + m[2][0]) / s;
    q.y = (m[1][2] + m[2][1]) / s;
    q.z = 0.25 * s;
  }
  return normalize(q);
}

@vertex
fn main_vertex(input: VertexInput) -> VertexOutput {
  var velocityMag = length(input.instanceVel.xyz);

  velocityMag = mix(0.01, 1.0, velocityMag);
  // use power to make velocityMag more extreme
  velocityMag = pow(velocityMag, 2.0);

  let scale = 0.04;
  let scaledPos = input.pos * vec4<f32>(velocityMag * 3.0, 0.01, 0.01, 1.0);

  // --- Rotation update: face velocity direction ---
  let v = input.instanceVel.xyz;
  let speed = length(v);
  var rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  if (speed > 0.0001) {
    let forward = normalize(v);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var right = cross(up, forward);
    if (length(right) < 0.0001) {
      right = vec3<f32>(1.0, 0.0, 0.0);
    } else {
      right = normalize(right);
    }
    let realUp = cross(forward, right);
    let rotMat = mat3x3<f32>(right, realUp, forward);
    rot = quat_from_matrix(rotMat);
  }

  let rotated = quat_rotate(rot, scaledPos.xyz);
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