export class ParticleBuffer {
  public positionBuffer: GPUBuffer;
  public velocityBuffer: GPUBuffer;
  public randomBuffer: GPUBuffer;
  public meshSampleBuffer: GPUBuffer; // Buffer for mesh surface samples
  public meshSampleCount: number;
  public particleCount: number;
  public setMeshSamples: (samples: Float32Array) => void;
  public agesBuffer: GPUBuffer;

  constructor(device: GPUDevice, particleCount: number, positions?: Float32Array, velocities?: Float32Array, random?: Float32Array) {
    this.particleCount = particleCount;
    // Create mesh sample buffer (for re-emission)
    // You should provide meshSamples externally as a Float32Array (count * 4 floats)
    // For now, create an empty buffer; fill it later when mesh samples are available
    this.meshSampleCount = 0;
    this.meshSampleBuffer = device.createBuffer({
      size: 4 * 4 * 1, // placeholder, will resize when mesh samples are set
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

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

    // Create inAges and outAges buffers (ping-pong)
    this.agesBuffer = device.createBuffer({
      size: particleCount * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Initialize ages to zero
    const ages = new Float32Array(particleCount * 4);
    ages.fill(0);
    device.queue.writeBuffer(this.agesBuffer, 0, ages);

    // Helper to set mesh samples later
    this.setMeshSamples = (samples: Float32Array) => {
      this.meshSampleCount = samples.length / 4;
      if (this.meshSampleBuffer) {
        this.meshSampleBuffer.destroy();
      }
      this.meshSampleBuffer = device.createBuffer({
        size: samples.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.meshSampleBuffer, 0, samples);
    };
  }

  // Static method for GPGPU update
  public updateParticles(
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    computePipeline: GPUComputePipeline,
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
        { binding: 0, resource: { buffer: this.positionBuffer } },
        { binding: 1, resource: { buffer: this.velocityBuffer } },
        { binding: 2, resource: { buffer: this.randomBuffer } },
        { binding: 3, resource: { buffer: this.agesBuffer } },
        { binding: 4, resource: { buffer: this.meshSampleBuffer } },
        { binding: 5, resource: { buffer: deltaTimeBuffer } },
        { binding: 6, resource: { buffer: uTimeBuffer } },
        { binding: 7, resource: { buffer: uRandomnessBuffer } },
        { binding: 8, resource: { buffer: uAirResistanceBuffer } },
        { binding: 9, resource: { buffer: uBoundaryRadiusBuffer } },
      ],
    });

    // Encode compute pass
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup);

    const WORKGROUP_SIZE = 64;
    const workgroupCount = Math.ceil(this.particleCount / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(workgroupCount);

    pass.end();
  }
}