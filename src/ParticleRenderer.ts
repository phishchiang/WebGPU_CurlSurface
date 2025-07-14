import particle_renderWGSL from './shaders/particle_render.wgsl?raw';
import { ParticleBuffer } from './ParticleBuffer';

export class ParticleRenderer {
  private pipeline: GPURenderPipeline;
  private viewMatrixBuffer: GPUBuffer;
  private projectionMatrixBuffer: GPUBuffer;
  private modelMatrixBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;

  // Add mesh buffers and layout
  private meshVertexBuffer: GPUBuffer;
  private meshIndexBuffer: GPUBuffer | undefined;
  private meshIndexCount: number;
  private meshVertexLayout: { arrayStride: number; attributes: GPUVertexAttribute[] };

  constructor(
    device: GPUDevice, 
    format: GPUTextureFormat,
    meshVertexBuffer: GPUBuffer,
    meshIndexBuffer: GPUBuffer | undefined,
    meshIndexCount: number,
    meshVertexLayout: { arrayStride: number; attributes: GPUVertexAttribute[] }
  ) {
    // Each mat4x4<f32> is 64 bytes
    this.projectionMatrixBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.viewMatrixBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.modelMatrixBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.meshVertexBuffer = meshVertexBuffer;
    this.meshIndexBuffer = meshIndexBuffer;
    this.meshIndexCount = meshIndexCount;
    this.meshVertexLayout = meshVertexLayout;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: device.createShaderModule({ code: particle_renderWGSL }),
        entryPoint: 'main_vertex',
        buffers: [
          // Mesh vertex buffer layout (from GLB)
          this.meshVertexLayout,
          {
            arrayStride: 16, // vec4<f32> position for each particle instance
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'float32x4' }, // instancePos
            ],
          },
          {
            arrayStride: 16, // vec4<f32> quaternion for each particle instance
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'float32x4' }, // instanceRot
            ],
          },
          {
            arrayStride: 16, // vec4<f32> quaternion for each particle instance
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 5, offset: 0, format: 'float32x4' }, // velocity
            ],
          },
        ],
      },
      fragment: {
        module: device.createShaderModule({ code: particle_renderWGSL }),
        entryPoint: 'main_fragment',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.projectionMatrixBuffer } },
        { binding: 1, resource: { buffer: this.viewMatrixBuffer } },
        { binding: 2, resource: { buffer: this.modelMatrixBuffer } },
      ],
    });
  }

  updateUniforms(
    device: GPUDevice,
    projectionMatrix: Float32Array,
    viewMatrix: Float32Array,
    modelMatrix: Float32Array
  ) {
    device.queue.writeBuffer(this.projectionMatrixBuffer, 0, projectionMatrix.buffer, projectionMatrix.byteOffset, 64);
    device.queue.writeBuffer(this.viewMatrixBuffer, 0, viewMatrix.buffer, viewMatrix.byteOffset, 64);
    device.queue.writeBuffer(this.modelMatrixBuffer, 0, modelMatrix.buffer, modelMatrix.byteOffset, 64);
  }

  render(pass: GPURenderPassEncoder, particleBuffer: ParticleBuffer) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);


    // Set mesh vertex buffer (slot 0)
    pass.setVertexBuffer(0, this.meshVertexBuffer);
    // Set instance position buffer (slot 1)
    pass.setVertexBuffer(1, particleBuffer.positionBuffer);
    // Set instance rotation buffer (slot 2)
    pass.setVertexBuffer(2, particleBuffer.rotationBuffer);
    // Set instance velocity buffer (slot 3)
    pass.setVertexBuffer(3, particleBuffer.velocityBuffer);
    // Set index buffer if available
    pass.setIndexBuffer(this.meshIndexBuffer!, 'uint16'); // or 'uint32' if needed
    pass.drawIndexed(this.meshIndexCount, particleBuffer.particleCount);
  }
}