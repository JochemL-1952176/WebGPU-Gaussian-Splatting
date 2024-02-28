import { StructuredView, makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import sharedShaderCode from './shaders/shared.wgsl?raw';
import rasterizeShaderCode from './shaders/rasterize.wgsl?raw';
import Scene from './scene';
import { Camera } from './cameraControls';
import GPUTimer from './GPUTimer';
import Sorter from './sorter';

export default class Renderer {
	cameraUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	controlsUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	gaussiansLayoutEntry: GPUBindGroupLayoutEntry;
	cameraUniforms: StructuredView;
	controlsUniforms: StructuredView;
	cameraUniformsBuffer: GPUBuffer;
	controlsUniformsBuffer: GPUBuffer;
	
	primaryRenderBindGroupLayout: GPUBindGroupLayout;
	#secondaryRenderBindGroup?: GPUBindGroup;
	
	#rasterShader: GPUShaderModule;
	#renderPipeline?: GPURenderPipeline;
	
	colorFormat: GPUTextureFormat;
	depthFormat: GPUTextureFormat = "depth32float";
	#canvasContext: GPUCanvasContext;
	#renderPassDescriptor: GPURenderPassDescriptor;
	#depthTextureDescriptor: GPUTextureDescriptor;
	#depthTexture: GPUTexture;
	
	#sorter?: Sorter

	constructor(device: GPUDevice, canvasContext: GPUCanvasContext) {
		this.#canvasContext = canvasContext;
		this.colorFormat = canvasContext.getCurrentTexture().format;

		this.cameraUniformsLayoutEntry = {
			binding: 0,
			visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
			buffer: { type: "uniform" }
		};

		this.gaussiansLayoutEntry = {
			binding: 1,
			visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" }
		}; 

		this.controlsUniformsLayoutEntry = {
			binding: 2,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "uniform" }
		};

		const rasterCode = sharedShaderCode + rasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });

		const shaderData = makeShaderDataDefinitions(rasterCode);
		this.cameraUniforms = makeStructuredView(shaderData.structs.Camera);
		this.controlsUniforms = makeStructuredView(shaderData.structs.RenderControls);

		this.controlsUniforms.set({ maxSH: 3, scaleMod: 1 });
		
		this.primaryRenderBindGroupLayout = device.createBindGroupLayout({
			label: "render bindgroup layout",
			entries: [
				this.cameraUniformsLayoutEntry,
				this.gaussiansLayoutEntry,
				this.controlsUniformsLayoutEntry
			]
		});

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

		device.queue.writeBuffer(this.controlsUniformsBuffer, 0, this.controlsUniforms.arrayBuffer);

		this.#depthTextureDescriptor = {
			size: [canvasContext.canvas.width, canvasContext.canvas.height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
			format: this.depthFormat
		};
		this.#depthTexture = device.createTexture(this.#depthTextureDescriptor);
		
		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: this.#canvasContext.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}],
			depthStencilAttachment: {
				view: this.#depthTexture!.createView(),
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store"
			}
		};

		this.setSize(device, canvasContext.canvas.width, canvasContext.canvas.height);
	}

	finalize(device: GPUDevice, scene: Scene) {
		this.#sorter?.destroy();
		this.#sorter = new Sorter(device, this, scene);
		
		const secondaryRenderBindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: "read-only-storage" }
			}]
		});

		this.#secondaryRenderBindGroup = device.createBindGroup({
			layout: secondaryRenderBindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: this.#sorter!.entryBufferA! }
			}]
		});

		const renderPipelineLayout = device.createPipelineLayout({
			label: "render pipeline layout",
			bindGroupLayouts: [this.primaryRenderBindGroupLayout, secondaryRenderBindGroupLayout]
		});
		
		this.#renderPipeline = device.createRenderPipeline({
			label: "render pipeline",
			primitive: { topology: 'triangle-strip' },
			layout: renderPipelineLayout,
			vertex: { module: this.#rasterShader, entryPoint: "vs" },
			fragment: {
				module: this.#rasterShader,
				entryPoint: "fs",
				targets: [{
					format: this.colorFormat,
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
				format: "depth32float",
				depthCompare: "less",
			}
		});
	}

	renderFrame(device: GPUDevice, scene: Scene, camera: Camera, timer?: GPUTimer) {
		console.assert(this.#sorter !== undefined, "Call finalize before rendering");
		if (this.#renderPipeline === undefined) throw new Error();

		if (camera.hasChanged) {		
			device.queue.writeBuffer(this.cameraUniformsBuffer, 0, camera.uniforms.arrayBuffer);
			camera.hasChanged = false;
		}

		const encoder = device.createCommandEncoder();
		// this.#sorter!.sort(encoder, scene, timer);

		this.#renderPassDescriptor.colorAttachments[0].view = this.#canvasContext.getCurrentTexture().createView();
		const renderPass =
			timer?.beginRenderPass(encoder, this.#renderPassDescriptor, 0) as GPURenderPassEncoder ??
			encoder.beginRenderPass(this.#renderPassDescriptor);
		
		renderPass.setPipeline(this.#renderPipeline);
		renderPass.setBindGroup(0, scene.renderBindGroup);
		renderPass.setBindGroup(1, this.#secondaryRenderBindGroup!);
		
		renderPass.draw(4, scene.splats.count);
		// renderPass.drawIndirect(this.#globalSortingBuffer!, this.#globalSortingBuffer!.size - 5 * Uint32Array.BYTES_PER_ELEMENT);
		
		renderPass.end();
		device.queue.submit([encoder.finish()]);
	}

	setSize(device: GPUDevice, width: number, height: number) {
		this.#depthTextureDescriptor.size = [width, height];
		this.#depthTexture.destroy();
		this.#depthTexture = device.createTexture(this.#depthTextureDescriptor);
		this.#renderPassDescriptor.depthStencilAttachment!.view = this.#depthTexture.createView();
	}

	destroy() {
		this.cameraUniformsBuffer.destroy();
		this.controlsUniformsBuffer.destroy();
		this.#sorter?.destroy();
		this.#depthTexture.destroy();
	}
}