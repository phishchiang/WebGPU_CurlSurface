export class ParticleBuffer {
  public positionBuffer: GPUBuffer;
  public velocityBuffer: GPUBuffer;
  public rotationBuffer: GPUBuffer;
  public randomBuffer: GPUBuffer;
  public particleCount: number;

  constructor(device: GPUDevice, particleCount: number, positions?: Float32Array, velocities?: Float32Array, random?: Float32Array) {
    this.particleCount = particleCount;

    // Create position buffer
    this.positionBuffer = device.createBuffer({
      size: particleCount * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // Initialize position buffer
    if (positions && positions.length === particleCount * 4) {
      device.queue.writeBuffer(this.positionBuffer, 0, positions);
    } else {
      // Fallback: random positions in a box
      const pos = new Float32Array(particleCount * 4);
      for (let i = 0; i < particleCount; ++i) {
        pos[i * 4 + 0] = (Math.random() - 0.5) * 4.0;
        pos[i * 4 + 1] = (Math.random() - 0.5) * 4.0;
        pos[i * 4 + 2] = (Math.random() - 0.5) * 4.0;
        pos[i * 4 + 3] = 1.0;
      }
      device.queue.writeBuffer(this.positionBuffer, 0, pos);
    }

    // Create velocity buffer
    this.velocityBuffer = device.createBuffer({
      size: particleCount * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    });

    // Create rotation buffer (quaternion: 4 floats per particle)
    this.rotationBuffer = device.createBuffer({
      size: particleCount * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // Initialize rotation buffer
    const rot = new Float32Array(particleCount * 4);
    for (let i = 0; i < particleCount; ++i) {
      // Identity quaternion (x=0, y=0, z=0, w=1)
      rot[i * 4 + 0] = 0.0;
      rot[i * 4 + 1] = 0.0;
      rot[i * 4 + 2] = 0.0;
      rot[i * 4 + 3] = 1.0;
    }
    device.queue.writeBuffer(this.rotationBuffer, 0, rot);

    // Initialize velocity buffer
    if (velocities && velocities.length === particleCount * 4) {
      device.queue.writeBuffer(this.velocityBuffer, 0, velocities);
    } else {
      // Fallback: zero velocities
      const vel = new Float32Array(particleCount * 4);
      for (let i = 0; i < particleCount; ++i) {
        vel[i * 4 + 0] = 0.0;
        vel[i * 4 + 1] = 0.0;
        vel[i * 4 + 2] = 0.0;
        vel[i * 4 + 3] = 0.0;
      }
      device.queue.writeBuffer(this.velocityBuffer, 0, vel);
    }

    // Create random buffer
    this.randomBuffer = device.createBuffer({
      size: particleCount * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Initialize random buffer
    if (random) {
      device.queue.writeBuffer(this.randomBuffer, 0, random);
    } else {
      const rand = new Float32Array(particleCount * 4);
      for (let i = 0; i < particleCount; ++i) {
        rand[i * 4 + 0] = Math.random();
        rand[i * 4 + 1] = Math.random();
        rand[i * 4 + 2] = Math.random();
        rand[i * 4 + 3] = 0.0;
      }
      device.queue.writeBuffer(this.randomBuffer, 0, rand);
    }
  }

  // Static method for GPGPU update
  static updateParticles(
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    computePipeline: GPUComputePipeline,
    inBuffer: ParticleBuffer,
    outBuffer: ParticleBuffer,
    deltaTime: number,
    deltaTimeBuffer: GPUBuffer,
    uTimeBuffer: GPUBuffer,
    uRandomnessBuffer: GPUBuffer,
    uAirResistanceBuffer: GPUBuffer,
    uBoundaryRadiusBuffer: GPUBuffer,
  ) {
    // Write deltaTime to buffer
    device.queue.writeBuffer(deltaTimeBuffer, 0, new Float32Array([deltaTime]).buffer);

    // Create bind group for this update
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuffer.positionBuffer } },
        { binding: 1, resource: { buffer: inBuffer.velocityBuffer } },
        { binding: 2, resource: { buffer: inBuffer.randomBuffer } },
        { binding: 3, resource: { buffer: inBuffer.rotationBuffer } },
        { binding: 4, resource: { buffer: outBuffer.positionBuffer } },
        { binding: 5, resource: { buffer: outBuffer.velocityBuffer } },
        { binding: 6, resource: { buffer: outBuffer.randomBuffer } },
        { binding: 7, resource: { buffer: outBuffer.rotationBuffer } },
        { binding: 8, resource: { buffer: deltaTimeBuffer } },
        { binding: 9, resource: { buffer: uTimeBuffer } },
        { binding: 10, resource: { buffer: uRandomnessBuffer } },
        { binding: 11, resource: { buffer: uAirResistanceBuffer } },
        { binding: 12, resource: { buffer: uBoundaryRadiusBuffer } },
      ],
    });

    // Encode compute pass
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup);

    const WORKGROUP_SIZE = 64;
    const workgroupCount = Math.ceil(inBuffer.particleCount / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(workgroupCount);

    pass.end();
  }
}