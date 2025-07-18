import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import basicWGSL from './shaders/basic.wgsl?raw'; // Raw String Import but only specific to Vite.
import { ArcballCamera, WASDCamera } from './camera';
import { createInputHandler } from './input';
import { loadAndProcessGLB } from './loadParseGLB';
import { RenderTarget } from './RenderTarget';
import { PostProcessEffect } from './postprocessing/PostProcessEffect';
import { PassThroughEffect } from './postprocessing/PassThroughEffect';
import { GrayscaleEffect } from './postprocessing/GrayscaleEffect';
import { FXAAEffect } from './postprocessing/FXAAEffect';
// Glow FX imports
import { BrightPassEffect } from './postprocessing/BrightPassEffect';
import { BlurEffect } from './postprocessing/GaussianBlurEffect';
import { GlowAddEffect } from './postprocessing/GlowAddEffect';
import { UnrealGlowEffect } from './postprocessing/UnrealGlowEffect';
import { ParticleBuffer } from './ParticleBuffer';
import { ParticleRenderer } from './ParticleRenderer';
import particle_computeWGSL from './shaders/particle_compute.wgsl?raw';

// const MESH_PATH = '/assets/meshes/light_color.glb';
// const MESH_PATH = '/assets/meshes/monkey.glb';
const MESH_PATH = '/assets/meshes/monkey_color.glb';
const PARTICLE_MESH_PATH = '/assets/meshes/cube.glb';
// arrayStride: For [x, y, z, nx, ny, nz, u, v] (stride = 8): vertexStride
const SAMPLED_MESH_VERTEX_STRIDE = 12; 
const PARTICLE_COUNT = 10000;

export class WebGPUApp{
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private presentationFormat!: GPUTextureFormat;
  private uniformBindGroup!: GPUBindGroup;
  private renderPassDescriptor!: GPURenderPassDescriptor;
  private cubeTexture!: GPUTexture;
  private cameras: { [key: string]: any };
  private aspect!: number;
  private params: { 
    type: 'arcball' | 'WASD'; 
    model: keyof typeof WebGPUApp.MODEL_PATHS;
    uTestValue: number; 
    uTestValue_02: number; 
    uNoiseScale: number; 
    uAirResistance: number; 
    uBoundaryRadius: number; 
    uGlow_Threshold: number;
    uGlow_ThresholdKnee: number; // Added for soft-knee threshold
    uGlow_Radius: number;
    uGlow_Intensity: number;
  } = {
    type: 'arcball',
    model: 'monkey',
    uTestValue: 1.0,
    uTestValue_02: 1.0,
    uNoiseScale: 2.0,
    uAirResistance: 0.8,
    uBoundaryRadius: 4.0,
    uGlow_Threshold: 0.5,
    uGlow_ThresholdKnee: 0.1,
    uGlow_Radius: 3.0,
    uGlow_Intensity: 0.5,
  };
  private uTime: number = 0.0;
  private gui: GUI;
  private lastFrameMS: number;
  private demoVerticesBuffer!: GPUBuffer;
  private loadVerticesBuffer!: GPUBuffer;
  private loadIndexBuffer!: GPUBuffer | undefined;
  private loadIndexCount!: number;
  private particleVerticesBuffer!: GPUBuffer;
  private particleIndexBuffer!: GPUBuffer | undefined;
  private particleIndexCount!: number;
  private particleVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[] };
  private uniformBuffer!: GPUBuffer;
  private viewMatrixBuffer!: GPUBuffer;
  private projectionMatrixBuffer!: GPUBuffer;
  private canvasSizeBuffer!: GPUBuffer;
  private uTimeBuffer!: GPUBuffer;
  private modelMatrixBuffer!: GPUBuffer;
  private uTestValueBuffer!: GPUBuffer;
  private uTestValue_02Buffer!: GPUBuffer;
  private uNoiseScaleBuffer!: GPUBuffer;
  private uAirResistanceBuffer!: GPUBuffer;
  private uBoundaryRadiusBuffer!: GPUBuffer;
  private loadVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[]; };
  private modelMatrix: Float32Array;
  private particle_modelMatrix: Float32Array;
  private viewMatrix: Float32Array;
  private projectionMatrix: Float32Array;
  private depthTexture!: GPUTexture;
  private sampler!: GPUSampler;
  private newCameraType!: string;
  private oldCameraType!: string;
  private renderTarget_ping!: RenderTarget;
  private renderTarget_pong!: RenderTarget;
  private postProcessEffects: PostProcessEffect[] = [];
  private inputHandler!: () => { 
    digital: { forward: boolean, backward: boolean, left: boolean, right: boolean, up: boolean, down: boolean, };
    analog: { x: number; y: number; zoom: number; touching: boolean };
  };
  private static readonly CLEAR_COLOR = [0.1, 0.1, 0.1, 1.0];
  private static readonly CAMERA_POSITION = vec3.create(1.2, 1.1, 2.3);
  private static readonly MODEL_PATHS = {
    monkey: '/assets/meshes/monkey_color.glb',
    teapot: '/assets/meshes/teapot.glb',
    cylinder: '/assets/meshes/light_color.glb',
  };
  private passThroughEffect!: PassThroughEffect;
  // Glow FX Variables
  private brightPassEffect!: BrightPassEffect;
  private blurEffectH!: BlurEffect;
  private blurEffectV!: BlurEffect;
  private glowAddEffect!: GlowAddEffect;
  private unrealGlowEffect!: UnrealGlowEffect;
  private enableGlow: boolean = true; // or control with GUI
  private particleRenderer!: ParticleRenderer;
  private particleBufferA!: ParticleBuffer;
  private particleBufferB!: ParticleBuffer;
  private usePing = true;
  private particleComputePipeline!: GPUComputePipeline;
  private deltaTimeBuffer!: GPUBuffer;
  private prevDt: number = 1.0 / 60.0; // Start with a default frame time (e.g., 60 FPS)
  private initialParticlePositions!: Float32Array;
  private initialParticleNormals!: Float32Array;
  private meshSamplesArray!: Float32Array;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gui = new GUI();
    this.cameras = {
      arcball: new ArcballCamera({ position: WebGPUApp.CAMERA_POSITION }),
      WASD: new WASDCamera({ position: WebGPUApp.CAMERA_POSITION }),
    };
    this.oldCameraType = this.params.type;
    this.lastFrameMS = Date.now();
    this.sampler = {} as GPUSampler;

     // The input handler
    this.inputHandler = createInputHandler(window, this.canvas);

    // Initialize matrices
    this.modelMatrix = mat4.identity();
    this.particle_modelMatrix = mat4.identity();
    this.viewMatrix = mat4.identity();
    this.projectionMatrix = mat4.identity();

    this.setupAndRender();
  }

  public async setupAndRender() {
    await this.initializeWebGPU();
    this.initRenderTargetsForPP();
    await this.initLoadAndProcessGLB();
    this.initUniformBuffer();
    await this.loadTexture();
    this.initParticleSystem();
    this.initCam();
    this.initPipelineBindGrp();
    this.initializeGUI();
    this.setupEventListeners();
    this.renderFrame();
  }

  private initParticleSystem() {
    // this.particleBufferA = new ParticleBuffer(this.device, PARTICLE_COUNT);
    // this.particleBufferB = new ParticleBuffer(this.device, PARTICLE_COUNT);

    this.particleBufferA = new ParticleBuffer(
      this.device,
      PARTICLE_COUNT,
      this.initialParticlePositions,
      undefined,
      undefined,
    );
    this.particleBufferB = new ParticleBuffer(
      this.device,
      PARTICLE_COUNT,
      this.initialParticlePositions,
      undefined,
      undefined,
    );

    this.particleBufferA.setMeshSamples(this.meshSamplesArray);
    this.particleBufferB.setMeshSamples(this.meshSamplesArray);

    // Pass the particle instancing mesh buffers and layout to the renderer
    this.particleRenderer = new ParticleRenderer(
      this.device,
      this.presentationFormat,
      this.particleVerticesBuffer,
      this.particleIndexBuffer,
      this.particleIndexCount,
      this.particleVertexLayout
    );

    // For some reason, it has to manually create the BindGroupLayout instead of usinng auto for layout
    const particleComputeBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, 
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, 
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    // Compute pipeline for particle update
    this.particleComputePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
            bindGroupLayouts: [particleComputeBindGroupLayout],
        }),
      compute: {
        module: this.device.createShaderModule({ code: particle_computeWGSL }),
        entryPoint: 'main',
      },
    });

    // Delta time uniform buffer
    this.deltaTimeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // Utility to sample N random points on mesh surface
  private sampleMeshSurfacePoints(
    vertexData: Float32Array,
    indices: Uint16Array,
    vertexStride: number, // e.g., 6 for [x,y,z,nx,ny,nz]
    positionOffset: number, // e.g., 0
    normalOffset: number,   // e.g., 3
    numParticles: number
  ): { position: Float32Array, normal: Float32Array }[] {
    const result: { position: Float32Array, normal: Float32Array }[] = [];
    const triangleCount = indices.length / 3;

    for (let i = 0; i < numParticles; i++) {
      // Pick a random triangle
      const triIdx = Math.floor(Math.random() * triangleCount);
      const i0 = indices[triIdx * 3 + 0];
      const i1 = indices[triIdx * 3 + 1];
      const i2 = indices[triIdx * 3 + 2];

      // Get vertex positions
      const v0 = vertexData.subarray(i0 * vertexStride + positionOffset, i0 * vertexStride + positionOffset + 3);
      const v1 = vertexData.subarray(i1 * vertexStride + positionOffset, i1 * vertexStride + positionOffset + 3);
      const v2 = vertexData.subarray(i2 * vertexStride + positionOffset, i2 * vertexStride + positionOffset + 3);

      // Get vertex normals
      const n0 = vertexData.subarray(i0 * vertexStride + normalOffset, i0 * vertexStride + normalOffset + 3);
      const n1 = vertexData.subarray(i1 * vertexStride + normalOffset, i1 * vertexStride + normalOffset + 3);
      const n2 = vertexData.subarray(i2 * vertexStride + normalOffset, i2 * vertexStride + normalOffset + 3);

      // Random barycentric coordinates
      let u = Math.random();
      let v = Math.random();
      if (u + v > 1) { u = 1 - u; v = 1 - v; }
      const w = 1 - u - v;

      // Interpolate position and normal
      const pos = [
        u * v0[0] + v * v1[0] + w * v2[0],
        u * v0[1] + v * v1[1] + w * v2[1],
        u * v0[2] + v * v1[2] + w * v2[2],
      ];
      const norm = [
        u * n0[0] + v * n1[0] + w * n2[0],
        u * n0[1] + v * n1[1] + w * n2[1],
        u * n0[2] + v * n1[2] + w * n2[2],
      ];
      // Normalize normal
      const normLen = Math.hypot(norm[0], norm[1], norm[2]);
      const normNormalized = [norm[0]/normLen, norm[1]/normLen, norm[2]/normLen];

      result.push({ position: new Float32Array(pos), normal: new Float32Array(normNormalized) });
    }
    return result;
  }

  private async initLoadAndProcessGLB() {
    const meshPath = WebGPUApp.MODEL_PATHS[this.params.model];
    const { interleavedData, indices, indexCount, vertexLayout } = await loadAndProcessGLB(meshPath);

    // Create vertex buffer
    const vertexBuffer = this.device.createBuffer({
      size: interleavedData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, // Add COPY_DST for writeBuffer
      mappedAtCreation: false, // Not needed
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, interleavedData);

    // Create index buffer if indices exist
    let indexBuffer: GPUBuffer | undefined = undefined;
    if (indices) {
      // Create index buffer
      // Pad index buffer size to next multiple of 4 for avoiding alignment issues
      // WebGPU requires buffer sizes to be a multiple of 4 bytes
      const paddedIndexBufferSize = Math.ceil(indices.byteLength / 4) * 4;

      indexBuffer = this.device.createBuffer({
        size: paddedIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint16Array(indexBuffer.getMappedRange()).set(indices);
      indexBuffer.unmap();
    }

    this.loadVerticesBuffer = vertexBuffer;
    this.loadIndexBuffer = indexBuffer;
    this.loadIndexCount = indexCount;
    this.loadVertexLayout = vertexLayout;
    console.log('Curl Surface Mesh :', this.loadVertexLayout);

    const positionOffset = 0;
    const normalOffset = 3;

    const sampledRandomSurfaceVertexArray = this.sampleMeshSurfacePoints(
      interleavedData,
      indices!,
      SAMPLED_MESH_VERTEX_STRIDE,
      positionOffset,
      normalOffset,
      PARTICLE_COUNT
    );

    this.meshSamplesArray = new Float32Array(PARTICLE_COUNT * 4);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.meshSamplesArray.set(sampledRandomSurfaceVertexArray[i].position, i * 4);
      this.meshSamplesArray[i * 4 + 3] = 1.0; // w
    }
    this.initialParticlePositions = new Float32Array(PARTICLE_COUNT * 4);
    this.initialParticleNormals = new Float32Array(PARTICLE_COUNT * 4);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const p = sampledRandomSurfaceVertexArray[i];
        this.initialParticlePositions.set(p.position, i * 4);
        this.initialParticlePositions[i * 4 + 3] = 1.0; // w
        // this.initialParticleNormals.set(p.normal, i * 4);
        // this.initialParticleNormals[i * 4 + 3] = 0.0; // w (unused)
      }
      
    // Load square mesh for particles
    const { 
      interleavedData: particleMeshData, 
      indices: particleMeshIndices, 
      indexCount: particleMeshIndexCount, 
      vertexLayout: particleMeshVertexLayout 
    } = await loadAndProcessGLB(PARTICLE_MESH_PATH);

    console.log('Particle Inatancing Mesh :', particleMeshVertexLayout);

    // Create vertex buffer for square mesh
    this.particleVerticesBuffer = this.device.createBuffer({
      size: particleMeshData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleVerticesBuffer.getMappedRange()).set(particleMeshData);
    this.particleVerticesBuffer.unmap();

    // Create index buffer for square mesh
    let particleMeshIndexBuffer: GPUBuffer | undefined = undefined;
    if (particleMeshIndices) {
      const paddedparticleMeshIndexBufferSize = Math.ceil(particleMeshIndices.byteLength / 4) * 4;
      particleMeshIndexBuffer = this.device.createBuffer({
        size: paddedparticleMeshIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint16Array(particleMeshIndexBuffer.getMappedRange()).set(particleMeshIndices);
      particleMeshIndexBuffer.unmap();
    }
    this.particleIndexBuffer = particleMeshIndexBuffer;
    this.particleIndexCount = particleMeshIndexCount;
    this.particleVertexLayout = particleMeshVertexLayout;
  }

  private initCam(){
    this.aspect = this.canvas.width / this.canvas.height;
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, this.aspect, 0.1, 1000.0);
    
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);
  }

  private async loadTexture() {
    const response = await fetch('../assets/img/uv1.png');
    const imageBitmap = await createImageBitmap(await response.blob());

    this.cubeTexture = this.device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: this.cubeTexture },
      [imageBitmap.width, imageBitmap.height]
    );
  }

  private initUniformBuffer() {
    // View Matrix
    this.viewMatrixBuffer = this.device.createBuffer({
      size: 16 * 4, // mat4x4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.viewMatrixBuffer, 0, this.viewMatrix.buffer);

    // Projection Matrix
    this.projectionMatrixBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);

    // Canvas Size
    this.canvasSizeBuffer = this.device.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const canvasSize = new Float32Array([this.canvas.width, this.canvas.height]);
    this.device.queue.writeBuffer(this.canvasSizeBuffer, 0, canvasSize.buffer);

    // uTime
    this.uTimeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTimeArr = new Float32Array([this.uTime]);
    this.device.queue.writeBuffer(this.uTimeBuffer, 0, uTimeArr.buffer);

    // Model Matrix
    this.modelMatrixBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.modelMatrixBuffer, 0, this.modelMatrix.buffer);

    // uTestValue
    this.uTestValueBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTestValueArr = new Float32Array([this.params.uTestValue]);
    this.device.queue.writeBuffer(this.uTestValueBuffer, 0, uTestValueArr.buffer);

    // uTestValue_02
    this.uTestValue_02Buffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTestValue_02Arr = new Float32Array([this.params.uTestValue_02]);
    this.device.queue.writeBuffer(this.uTestValue_02Buffer, 0, uTestValue_02Arr.buffer);

    // uTandomness
    this.uNoiseScaleBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uNoiseScaleArr = new Float32Array([this.params.uNoiseScale]);
    this.device.queue.writeBuffer(this.uNoiseScaleBuffer, 0, uNoiseScaleArr.buffer);

    // uAirResistance
    this.uAirResistanceBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uAirResistanceArr = new Float32Array([this.params.uAirResistance]);
    this.device.queue.writeBuffer(this.uAirResistanceBuffer, 0, uAirResistanceArr.buffer);

    // uBoundaryRadius
    this.uBoundaryRadiusBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uBoundaryRadiusArr = new Float32Array([this.params.uBoundaryRadius]);
    this.device.queue.writeBuffer(this.uBoundaryRadiusBuffer, 0, uBoundaryRadiusArr.buffer);
  }

  private setupEventListeners() {
    window.addEventListener('resize', this.resize.bind(this));
  }

  private resize() {
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.aspect = this.canvas.width / this.canvas.height;
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, this.aspect, 1, 100.0);
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
    });

    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);

    // CanvasSize vec2f 2 value 8 bytes, index 48
    const canvasSizeArray = new Float32Array([this.canvas.width, this.canvas.height]);
    this.device.queue.writeBuffer(this.canvasSizeBuffer, 0, canvasSizeArray.buffer);

    // Recreate the depth texture to match the new canvas size
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Resize the render targets
    this.renderTarget_ping.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);
    this.renderTarget_pong.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);

  }

  private initializeGUI() {
    // this.gui.add(this.params, 'type', ['arcball', 'WASD']).onChange(() => {
    //   this.newCameraType = this.params.type;
    //   this.cameras[this.newCameraType].matrix = this.cameras[this.oldCameraType].matrix;
    //   this.oldCameraType = this.newCameraType
    // });

    this.gui.add(this.params, 'model', ['monkey', 'teapot', 'cylinder']).onChange(async () => {
      await this.initLoadAndProcessGLB(); // Reload mesh and reinitialize as needed
      this.initParticleSystem(); // Re-init particle system if needed
    });
    
    // this.gui.add(this.params, 'uTestValue', 0.0, 1.0).step(0.01).onChange((value) => {
    //   this.updateFloatUniform( 'uTestValue', value );
    // });
    // this.gui.add(this.params, 'uTestValue_02', 0.0, 1.0).step(0.01).onChange((value) => {
    //   this.updateFloatUniform( 'uTestValue_02', value );
    // });

    const particleFolder = this.gui.addFolder('Particle Params');
    particleFolder.add(this.params, 'uNoiseScale', 0.0, 5.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uNoiseScale', value );
    });
    particleFolder.add(this.params, 'uAirResistance', 0.0, 1.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uAirResistance', value );
    });
    particleFolder.add(this.params, 'uBoundaryRadius', 0.5, 10.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uBoundaryRadius', value );
    });
    particleFolder.open();
    
    const glowFolder = this.gui.addFolder('Glow FX');
    glowFolder.add(this.params, 'uGlow_Threshold', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_ThresholdKnee', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Radius', 0.1, 20.0).step(0.1).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Intensity', 0.0, 1.0).step(0.001).onChange(() => this.updateGlowUniforms());
    // glowFolder.open();
  }

  private updateGlowUniforms() {
    this.brightPassEffect.setThreshold(this.params.uGlow_Threshold);
    this.brightPassEffect.setKnee(this.params.uGlow_ThresholdKnee);
    this.blurEffectH.setRadius(this.params.uGlow_Radius);
    this.blurEffectV.setRadius(this.params.uGlow_Radius);
    this.glowAddEffect.setIntensity(this.params.uGlow_Intensity);
  }

  private updateFloatUniform(key: keyof typeof this.params, value: number) {
    const updatedFloatArray = new Float32Array([value]);
    switch (key) {
      case 'uTestValue':
        this.device.queue.writeBuffer(this.uTestValueBuffer, 0, updatedFloatArray.buffer);
        break;
      case 'uTestValue_02':
        this.device.queue.writeBuffer(this.uTestValue_02Buffer, 0, updatedFloatArray.buffer);
        break;
      case 'uNoiseScale':
        this.device.queue.writeBuffer(this.uNoiseScaleBuffer, 0, updatedFloatArray.buffer);
        break;
      case 'uAirResistance':
        this.device.queue.writeBuffer(this.uAirResistanceBuffer, 0, updatedFloatArray.buffer);
        break;
      case 'uBoundaryRadius':
        this.device.queue.writeBuffer(this.uBoundaryRadiusBuffer, 0, updatedFloatArray.buffer);
        break;
      default:
        console.error(`Unknown key: ${key}`);
        return;
    }
  }

  private async initializeWebGPU() {
    const adapter = await navigator.gpu?.requestAdapter({ featureLevel: 'compatibility' });
    this.device = await adapter?.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
    }) as GPUDevice;

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    });

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned later
          clearValue: WebGPUApp.CLEAR_COLOR,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ] as Iterable< GPURenderPassColorAttachment | null | undefined>,
      depthStencilAttachment: {
        view: this.depthTexture.createView(), // Assign a valid GPUTextureView
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };
  }

  private initPipelineBindGrp() {

    const uniformBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // viewMatrix
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // projectionMatrix
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // canvasSize
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTime
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // modelMatrix
        { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTestValue
        { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTestValue_02
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }, // Sampler
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // Texture
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [uniformBindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: basicWGSL }),
        entryPoint: 'vertex_main',
        buffers: [{
          arrayStride: this.loadVertexLayout.arrayStride,
          attributes: this.loadVertexLayout.attributes,
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: basicWGSL }),
        entryPoint: 'fragment_main',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.uniformBindGroup = this.device.createBindGroup({
      layout: uniformBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.viewMatrixBuffer } },
        { binding: 1, resource: { buffer: this.projectionMatrixBuffer } },
        { binding: 2, resource: { buffer: this.canvasSizeBuffer } },
        { binding: 3, resource: { buffer: this.uTimeBuffer } },
        { binding: 4, resource: { buffer: this.modelMatrixBuffer } },
        { binding: 5, resource: { buffer: this.uTestValueBuffer } },
        { binding: 6, resource: { buffer: this.uTestValue_02Buffer } },
        { binding: 7, resource: this.sampler },
        { binding: 8, resource: this.cubeTexture.createView() },
      ],
    });
  }

  private getViewMatrix(deltaTime: number) {
    const camera = this.cameras[this.params.type];
    const viewMatrix =  camera.update(deltaTime, this.inputHandler());
    return viewMatrix;
  }

  private initRenderTargetsForPP() {
    // Create ping-pong render targets
    this.renderTarget_ping = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );
    this.renderTarget_pong = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );

    // Init useful pass-through effect 
    this.passThroughEffect = new PassThroughEffect(this.device, this.presentationFormat, this.sampler);

    this.brightPassEffect = new BrightPassEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Threshold, this.params.uGlow_ThresholdKnee);
    // Add post-processing effects
    this.postProcessEffects.push(
      // new GrayscaleEffect(this.device, this.presentationFormat, this.sampler),
      // this.brightPassEffect,
      new FXAAEffect(this.device, this.presentationFormat, this.sampler, [this.canvas.width, this.canvas.height]),
    );

    this.blurEffectH = new BlurEffect(this.device, this.presentationFormat, this.sampler, [1.0, 0.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.blurEffectV = new BlurEffect(this.device, this.presentationFormat, this.sampler, [0.0, 1.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.glowAddEffect = new GlowAddEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Intensity );
    this.unrealGlowEffect = new UnrealGlowEffect(
      this.device,
      this.presentationFormat,
      this.sampler,
      this.canvas.width,
      this.canvas.height,
      4, // levels, adjust as needed
      this.brightPassEffect,
      this.blurEffectH,
      this.blurEffectV,
      this.glowAddEffect,
      this.passThroughEffect
    );
  }

  private renderFrame() {
    const now = Date.now();
 
    // Clamp and smooth uDeltaTime on the CPU side before passing to WGSL.
    const deltaTime = (now - this.lastFrameMS) / 1000;
    const minDt = 1.0 / 120.0;
    const maxDt = 1.0 / 30.0;
    const clampedDt = Math.max(minDt, Math.min(maxDt, deltaTime));
    let smoothedDt = this.prevDt * 0.8 + clampedDt * 0.2;
    this.prevDt = smoothedDt;

    this.lastFrameMS = now;

    // Update the uniform uTime value
    this.uTime += this.prevDt;
    const uTimeFloatArray = new Float32Array([this.uTime]);
    this.device.queue.writeBuffer(this.uTimeBuffer, 0, uTimeFloatArray.buffer);

    this.viewMatrix = this.getViewMatrix(this.prevDt);
    this.device.queue.writeBuffer(this.viewMatrixBuffer, 0, this.viewMatrix.buffer);

    // Set up a render pass target based on post-processing effects
    if (this.postProcessEffects.length === 0) {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context.getCurrentTexture().createView();
    } else {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.renderTarget_ping.view;
    }

    // Update the depth attachment view
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthTexture.createView();

    const commandEncoder = this.device.createCommandEncoder();

    // Select ping or pong buffers
    const inBuffer = this.usePing ? this.particleBufferA : this.particleBufferB;
    const outBuffer = this.usePing ? this.particleBufferB : this.particleBufferA;

    // --- Update particles on GPU ---
    ParticleBuffer.updateParticles(
      this.device,
      commandEncoder,
      this.particleComputePipeline,
      inBuffer,
      outBuffer,
      this.prevDt,
      this.deltaTimeBuffer,
      this.uTimeBuffer,
      this.uNoiseScaleBuffer,
      this.uAirResistanceBuffer,
      this.uBoundaryRadiusBuffer,
    );

    // --- Render scene ---
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    // passEncoder.setPipeline(this.pipeline);
    // passEncoder.setBindGroup(0, this.uniformBindGroup);
    // passEncoder.setVertexBuffer(0, this.loadVerticesBuffer);
    // passEncoder.setIndexBuffer(this.loadIndexBuffer!, 'uint16');
    // passEncoder.drawIndexed(this.loadIndexCount);

    this.particleRenderer.updateUniforms(this.device, this.projectionMatrix, this.viewMatrix, this.particle_modelMatrix);
    // Render using the *output* buffer (the one just written to)
    this.particleRenderer.render(passEncoder, outBuffer);

    passEncoder.end();

    // Swap for next frame
    this.usePing = !this.usePing;

    // Apply post-processing effects if any
    let finalOutputView = this.renderTarget_ping.view;
    if (this.postProcessEffects.length > 0) {
      let inputView = this.renderTarget_ping.view;
      let outputView = this.renderTarget_pong.view;
      for (let i = 0; i < this.postProcessEffects.length; i++) {
        const isLast = i === this.postProcessEffects.length - 1;

        if(!this.enableGlow) { // Only use single output for PostProcessEffects
          finalOutputView = isLast ? this.context.getCurrentTexture().createView() : outputView;
        } else { // Make sure to continue using ping-pong buffers when applying glowFX afterwards
          finalOutputView = outputView;
        }
        
        this.postProcessEffects[i].apply(
          commandEncoder,
          { A: inputView },
          finalOutputView,
          [this.canvas.width, this.canvas.height]
        );
        if (!isLast) {
          [inputView, outputView] = [outputView, inputView];
        }
      }
      if (this.enableGlow) {
        this.unrealGlowEffect.apply(
          commandEncoder,
          finalOutputView,
          this.context.getCurrentTexture().createView()
        );
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(this.renderFrame.bind(this));
  }
}

const app = new WebGPUApp(document.getElementById('app') as HTMLCanvasElement);