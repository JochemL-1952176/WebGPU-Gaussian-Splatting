import { StructuredView, makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import shaderCode from './shaders/gsplat.wgsl?raw';
import GPUTimer from './GPUTimer';
import Scene from './scene';

export default class Renderer {
	cameraUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	cameraUniforms: StructuredView;
	cameraUniformsBuffer: GPUBuffer;
	controlsUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	controlsUniforms: StructuredView;
	controlsUniformsBuffer: GPUBuffer
	gaussiansLayoutEntry: GPUBindGroupLayoutEntry;
	renderBindGroupLayout: GPUBindGroupLayout;

	renderPipelineDescriptor: GPURenderPipelineDescriptor;
	renderPipeline: GPURenderPipeline;

	canvasFormat: GPUTextureFormat;
	#canvasContext: GPUCanvasContext;
	#renderPassDescriptor: GPURenderPassDescriptor;
	#depthTextureDescriptor: GPUTextureDescriptor;
	#depthTexture?: GPUTexture;

	gpuTimer: GPUTimer;

	constructor(device: GPUDevice, canvasContext: GPUCanvasContext, canvasFormat: GPUTextureFormat) {
		this.#canvasContext = canvasContext;
		this.canvasFormat = canvasFormat;
		this.gpuTimer = new GPUTimer(device, 1);

		const shaderModule = device.createShaderModule({ code: shaderCode });
		const shaderData = makeShaderDataDefinitions(shaderCode);
		this.cameraUniforms = makeStructuredView(shaderData.structs.Camera);
		this.controlsUniforms = makeStructuredView(shaderData.structs.Controls);
		
		this.cameraUniformsLayoutEntry = {
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "uniform" }
		};

		this.controlsUniformsLayoutEntry = {
			binding: 1,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" }
		};

		this.gaussiansLayoutEntry = {
			binding: 2,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" }
		}; 

		this.renderBindGroupLayout = device.createBindGroupLayout({
			label: "render bindgroup layout",
			entries: [
				this.cameraUniformsLayoutEntry, 
				this.controlsUniformsLayoutEntry,
				this.gaussiansLayoutEntry
			]
		});

		const renderPipelineLayout = device.createPipelineLayout({
			label: "render pipeline layout",
			bindGroupLayouts: [this.renderBindGroupLayout]
		});

		this.renderPipelineDescriptor = {
			label: "render pipeline",
			primitive: { topology: 'triangle-strip' },
			layout: renderPipelineLayout,
			vertex: { module: shaderModule, entryPoint: "vs" },
			fragment: {
				module: shaderModule,
				entryPoint: "fs",
				targets: [{
					format: canvasFormat,
					blend: {
						alpha: {
							srcFactor: "one",
							dstFactor: "one-minus-src-alpha"
						},
						color: {
							srcFactor: "src-alpha",
							dstFactor: "one-minus-src-alpha"
						}
					}
				}]
			},
			depthStencil: {
				depthWriteEnabled: true,
				format: "depth16unorm",
				depthCompare: "less",
			}
		};
		
		this.renderPipeline = device.createRenderPipeline(this.renderPipelineDescriptor);

		this.cameraUniformsBuffer = device.createBuffer({
			label: "Camera uniforms buffer",
			size: this.cameraUniforms.arrayBuffer.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		
		this.controlsUniformsBuffer = device.createBuffer({
			label: "Controls uniforms buffer",
			size: this.controlsUniforms.arrayBuffer.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		this.#depthTextureDescriptor = {
			size: [canvasContext.canvas.width, canvasContext.canvas.height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
			format: this.renderPipelineDescriptor.depthStencil!.format!
		};
		
		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: this.#canvasContext.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}]
		};

		this.setSize(device, canvasContext.canvas.width, canvasContext.canvas.height);
	}

	renderFrame(device: GPUDevice, scene: Scene) {
		const encoder = device.createCommandEncoder();
		this.#renderPassDescriptor.colorAttachments[0].view = this.#canvasContext.getCurrentTexture().createView();
		const renderPass = this.gpuTimer.beginRenderPass(encoder, this.#renderPassDescriptor, 0);
		
		renderPass.executeBundles([scene.renderBundle])
		
		renderPass.end();
		device.queue.submit([encoder.finish()]);
	}

	setSize(device: GPUDevice, width: number, height: number) {
		this.#depthTextureDescriptor.size = [width, height];
		this.#depthTexture?.destroy();
		this.#depthTexture = device.createTexture(this.#depthTextureDescriptor);
		this.#renderPassDescriptor.depthStencilAttachment = {
			view: this.#depthTexture!.createView(),
			depthClearValue: 1.0,
			depthLoadOp: "clear",
			depthStoreOp: "store"
		}
	}
}