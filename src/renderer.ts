import { StructuredView, makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import sharedShaderCode from './shaders/shared.wgsl?raw';
import rasterizeShaderCode from './shaders/rasterize.wgsl?raw';
import Scene from './scene';
import { Camera } from './cameraControls';
import GPUTimer from './GPUTimer';
import SplatSorter from './sorter';

export type RenderTimings = {
	sorting: number,
	rendering: number
};

export default class Renderer {
	cameraUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	controlsUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	gaussiansLayoutEntry: GPUBindGroupLayoutEntry;
	cameraUniforms: StructuredView;
	controlsUniforms: StructuredView;
	cameraUniformsBuffer: GPUBuffer;
	controlsUniformsBuffer: GPUBuffer;
	
	#timer: GPUTimer;
	#colorFormat: GPUTextureFormat;
	#canvasContext: GPUCanvasContext;
	#rasterShader: GPUShaderModule;
	#renderPassDescriptor: GPURenderPassDescriptor;
	primaryRenderBindGroupLayout: GPUBindGroupLayout;
	
	#sorter?: SplatSorter;
	#renderBundle?: GPURenderBundle;
	
	constructor(device: GPUDevice, canvasContext: GPUCanvasContext) {
		this.#canvasContext = canvasContext;
		this.#colorFormat = canvasContext.getCurrentTexture().format;

		this.#timer = new GPUTimer(device, 6);

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

	finalize(device: GPUDevice, scene: Scene) {
		this.#sorter?.destroy();
		this.#sorter = new SplatSorter(device, this, scene.splats);
		
		const secondaryRenderBindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: "read-only-storage" }
			}]
		});

		const secondaryRenderBindGroup = device.createBindGroup({
			layout: secondaryRenderBindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: this.#sorter!.entryBufferA }
			}]
		});

		const renderPipelineLayout = device.createPipelineLayout({
			label: "render pipeline layout",
			bindGroupLayouts: [this.primaryRenderBindGroupLayout, secondaryRenderBindGroupLayout]
		});
		
		const renderPipeline = device.createRenderPipeline({
			label: "render pipeline",
			primitive: { topology: 'triangle-strip' },
			layout: renderPipelineLayout,
			vertex: { module: this.#rasterShader, entryPoint: "vs" },
			fragment: {
				module: this.#rasterShader,
				entryPoint: "fs",
				targets: [{
					format: this.#colorFormat,
					// dst.color = src.color * src.alpha + dst.color * (1 - src.alpha)
					blend: {
						alpha: {
							srcFactor: "one-minus-dst-alpha",
							dstFactor: "one"
						},
						color: {
							srcFactor: "one-minus-dst-alpha",
							dstFactor: "one"
						}
					}
				}]
			}
		});

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.#colorFormat]
		});

		renderPassEncoder.setPipeline(renderPipeline);
		renderPassEncoder.setBindGroup(0, scene.renderBindGroup);
		renderPassEncoder.setBindGroup(1, secondaryRenderBindGroup);
		renderPassEncoder.drawIndirect(this.#sorter.drawIndirectBuffer, 0);

		this.#renderBundle = renderPassEncoder.finish();
	}

	renderFrame(device: GPUDevice, scene: Scene, camera: Camera) {
		const finalized = this.#sorter !== undefined && this.#renderBundle !== undefined;
		console.assert(finalized, "Call finalize before rendering");
		if (!finalized) throw new Error();
		const encoder = device.createCommandEncoder();
		
		if (camera.hasChanged) {		
			device.queue.writeBuffer(this.cameraUniformsBuffer, 0, camera.uniforms.arrayBuffer);
			camera.hasChanged = false;
		}

		this.#sorter!.sort(encoder, scene, this.#timer);
		
		this.#renderPassDescriptor.colorAttachments[0].view = this.#canvasContext.getCurrentTexture().createView();

		const renderPass = this.#timer.beginRenderPass(encoder, this.#renderPassDescriptor, 5)
		renderPass.executeBundles([this.#renderBundle!]);
		renderPass.end();

		device.queue.submit([encoder.finish()]);
	}

	setSize(_device: GPUDevice, _width: number, _height: number) {}

	timings: RenderTimings = { sorting: 0, rendering: 0 };
	async getTimings() {
		const results = await this.#timer.getResults();
		if (results) {
			this.timings.sorting = results
				.slice(0, results.length - 1)
				.reduce((acc: number, v: number) => acc + v)

			this.timings.rendering = results[results.length - 1]
		}

		return this.timings;
	}

	destroy() {
		this.cameraUniformsBuffer.destroy();
		this.controlsUniformsBuffer.destroy();
		this.#sorter?.destroy();
		this.#timer.destroy();
	}
}