@group(0) @binding(0) var<storage, read>  inPositions  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outPositions  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  inVelocities : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outVelocities : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  inRandom : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>  inMeshSamples : array<vec4<f32>>;
@group(0) @binding(6) var<uniform> uDeltaTime : f32;
@group(0) @binding(7) var<uniform> uTime : f32;
@group(0) @binding(8) var<uniform> uRandomness : f32;
@group(0) @binding(9) var<uniform> uAirResistance : f32;
@group(0) @binding(10) var<uniform> uBoundaryRadius : f32;

fn mod289_vec3(x: vec3<f32>) -> vec3<f32> {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x: vec4<f32>) -> vec4<f32> {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
  return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {
  return 1.79284291400159 - 0.85373472095314 * r;
}

fn snoise(v: vec3<f32>) -> f32 {
  let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
  let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

  var i = floor(v + dot(v, C.yyy));
  let x0 = v - i + dot(i, C.xxx);

  let g = step(x0.yzx, x0.xyz);
  let l = 1.0 - g;
  let i1 = min(g.xyz, l.zxy);
  let i2 = max(g.xyz, l.zxy);

  let x1 = x0 - i1 + C.xxx;
  let x2 = x0 - i2 + 2.0 * C.xxx;
  let x3 = x0 - 1.0 + 3.0 * C.xxx;

  i = mod289_vec3(i);

  let p = permute(
    permute(
      permute(vec4<f32>(i.z) + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + vec4<f32>(i.y) + vec4<f32>(0.0, i1.y, i2.y, 1.0)
    )
    + vec4<f32>(i.x) + vec4<f32>(0.0, i1.x, i2.x, 1.0)
  );

  let n_ = 1.0 / 7.0;
  let ns = n_ * D.wyz - D.xzx;

  let j = p - 49.0 * floor(p * ns.z * ns.z);

  let x_ = floor(j * ns.z);
  let y_ = floor(j - 7.0 * x_);

  let x = x_ * ns.x + ns.yyyy;
  let y = y_ * ns.x + ns.yyyy;
  let h = 1.0 - abs(x) - abs(y);

  let b0 = vec4<f32>(x.xy, y.xy);
  let b1 = vec4<f32>(x.zw, y.zw);

  let s0 = floor(b0) * 2.0 + 1.0;
  let s1 = floor(b1) * 2.0 + 1.0;
  let sh = -step(h, vec4<f32>(0.0));

  let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  let a1 = b1.xzyw + s1.xzyw * sh.zzww;

  var p0 = vec3<f32>(a0.xy, h.x);
  var p1 = vec3<f32>(a0.zw, h.y);
  var p2 = vec3<f32>(a1.xy, h.z);
  var p3 = vec3<f32>(a1.zw, h.w);

  let norm = taylorInvSqrt(vec4<f32>(
    dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)
  ));
  p0 = p0 * norm.x;
  p1 = p1 * norm.y;
  p2 = p2 * norm.z;
  p3 = p3 * norm.w;

  var m = max(0.6 - vec4<f32>(
    dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)
  ), vec4<f32>(0.0));
  m = m * m;

  return 42.0 * dot(
    m * m,
    vec4<f32>(
      dot(p0, x0),
      dot(p1, x1),
      dot(p2, x2),
      dot(p3, x3)
    )
  );
}

fn snoiseVec3(x: vec3<f32>) -> vec3<f32> {
  let s = snoise(x);
  let s1 = snoise(vec3<f32>(x.y - 19.1, x.z + 33.4, x.x + 47.2));
  let s2 = snoise(vec3<f32>(x.z + 74.2, x.x - 124.5, x.y + 99.4));
  return vec3<f32>(s, s1, s2);
}

fn curlNoise(p: vec3<f32>) -> vec3<f32> {
  let e = 0.1;
  let dx = vec3<f32>(e, 0.0, 0.0);
  let dy = vec3<f32>(0.0, e, 0.0);
  let dz = vec3<f32>(0.0, 0.0, e);

  let p_x0 = snoiseVec3(p - dx);
  let p_x1 = snoiseVec3(p + dx);
  let p_y0 = snoiseVec3(p - dy);
  let p_y1 = snoiseVec3(p + dy);
  let p_z0 = snoiseVec3(p - dz);
  let p_z1 = snoiseVec3(p + dz);

  let x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  let y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  let z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  let divisor = 1.0 / (2.0 * e);
  return normalize(vec3<f32>(x, y, z) * divisor);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;

  // Simple Euler integration
  var pos = inPositions[idx];
  var vel = inVelocities[idx];
  var ran = inRandom[idx];

  // Set acceleration
  var acceleration = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let noise_val = curlNoise(pos.xyz * 0.5 * uRandomness + vec3<f32>(uTime * 0.04 * uRandomness));
  acceleration = acceleration + vec4<f32>(noise_val, 0.0);

  // set boundary
  let dist_to_center = length(pos - vec4<f32>(0.0, 0.0, 0.0, 1.0));
  let boundary_dir = -normalize(pos - vec4<f32>(0.0, 0.0, 0.0, 1.0));
  let boundary_force = smoothstep(uBoundaryRadius * 0.75, uBoundaryRadius, dist_to_center);
  acceleration = acceleration + boundary_dir * boundary_force * 1.0;

  // Set velocity
  vel = vel + acceleration * 10.0 * uDeltaTime;
  let velocity_random = mix(1.0, 8.0, ran.x);
  vel = vel * (1.0 - mix(0.05, 1.0, uAirResistance));
  
  // Set position
  pos = pos + vel * velocity_random * uDeltaTime;

  // After updating pos
  let mesh_center = vec3<f32>(0.0, 0.0, 0.0); // or your mesh's center
  let max_distance = uBoundaryRadius * 0.3; // tweak as needed
  if (length(pos.xyz - mesh_center) > max_distance) {
    // Reset position to a random mesh sample
    let meshSampleCount = arrayLength(&inMeshSamples);
    let randomIdx = u32(abs(fract(ran.y) * f32(meshSampleCount)));
    pos = inMeshSamples[randomIdx];
    vel = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  outPositions[idx] = pos;
  outVelocities[idx] = vel;
}