import { StructuredView, makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import sharedShaderCode from './shaders/shared.wgsl?raw';
import rasterizeShaderCode from './shaders/rasterize.wgsl?raw';
import Scene from './scene';
import { Camera } from './cameraControls';
import GPUTimer from './GPUTimer';
import SplatSorter from './sorter';
import { BindingApi, FolderApi } from '@tweakpane/core';
import { Pane } from 'tweakpane';

export abstract class Renderer {
	common: CommonRendererData;

	constructor(common: CommonRendererData) {
		this.common = common;
	}

	abstract finalize(device: GPUDevice, scene: Scene): void;
	abstract renderFrame(device: GPUDevice, scene: Scene, camera: Camera): void;
	abstract setSize(device: GPUDevice, width: number, height: number): void;
	abstract telemetryPanes(root: FolderApi | Pane, interval: number): void;
	abstract destroy(): void;
}

export class SortingRenderer extends Renderer {

	#rasterShader: GPUShaderModule;
	#renderPassDescriptor: GPURenderPassDescriptor;
	
	#timer: GPUTimer;
	#sorter?: SplatSorter;
	#renderBundle?: GPURenderBundle;

	#telemetry = {
		sorting: 0,
		rendering: 0
	};

	#sortingTelemetryBinding?: BindingApi<unknown, number>;
	#renderingTelemetryBinding?: BindingApi<unknown, number>;
	
	constructor(device: GPUDevice, common: CommonRendererData) {
		super(common);

		this.#timer = new GPUTimer(device, 6);

		const rasterCode = sharedShaderCode + rasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });
		
		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: this.common.canvasContext.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}]
		};

		this.setSize(device, this.common.canvasContext.canvas.width, this.common.canvasContext.canvas.height);
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
			bindGroupLayouts: [this.common.primaryRenderBindGroupLayout, secondaryRenderBindGroupLayout]
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
					format: this.common.canvasContext.getCurrentTexture().format,
					blend: {
						alpha: {
							srcFactor: "src-alpha",
							dstFactor: "one-minus-src-alpha"
						},
						color: {
							srcFactor: "src-alpha",
							dstFactor: "one-minus-src-alpha"
						}
					}
				}]
			}
		});

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.common.canvasContext.getCurrentTexture().format]
		});

		renderPassEncoder.setPipeline(renderPipeline);
		renderPassEncoder.setBindGroup(0, scene.renderBindGroup);
		renderPassEncoder.setBindGroup(1, secondaryRenderBindGroup);
		renderPassEncoder.drawIndirect(this.#sorter.drawIndirectBuffer, 0);

		this.#renderBundle = renderPassEncoder.finish();
	}

	renderFrame(device: GPUDevice, scene: Scene, camera: Camera) {
		console.assert(this.#sorter !== undefined && this.#renderBundle !== undefined, "Call finalize before rendering");
		const encoder = device.createCommandEncoder();
		
		if (camera.hasChanged) {		
			device.queue.writeBuffer(this.common.cameraUniformsBuffer, 0, camera.uniforms.arrayBuffer);
			camera.hasChanged = false;
		}

		this.#sorter!.sort(encoder, scene, this.#timer);
		
		this.#renderPassDescriptor.colorAttachments[0].view = this.common.canvasContext.getCurrentTexture().createView();

		const renderPass = this.#timer.beginRenderPass(encoder, this.#renderPassDescriptor, 5)
		renderPass.executeBundles([this.#renderBundle!]);
		renderPass.end();

		device.queue.submit([encoder.finish()]);
		this.#getTimings();
	}

	setSize(_device: GPUDevice, _width: number, _height: number) {}

	telemetryPanes(root: FolderApi | Pane, interval: number = 50) {
		this.#sortingTelemetryBinding = root.addBinding(this.#telemetry, 'sorting', {
			readonly: true,
			view: 'graph',
			label: 'Sort time',
			format: (v: number) => `${v.toFixed(2)}μs`,
			min: 0, max: 100_000 / 3,
			interval: interval
		});
		
		this.#renderingTelemetryBinding = root.addBinding(this.#telemetry, 'rendering', {
			readonly: true,
			view: 'graph',
			label: 'Render time',
			format: (v: number) => `${v.toFixed(2)}μs`,
			min: 0, max: 100_000 / 3,
			interval: interval
		});
	}

	#getTimings() {
		this.#timer.getResults().then((results?: Array<number>) => {
			if (results) {
				this.#telemetry.sorting = results
				.slice(0, results.length - 1)
				.reduce((acc: number, v: number) => acc + v);

				this.#telemetry.rendering = results[results.length - 1]
			}
		});
	}

	destroy() {
		this.#sorter?.destroy();
		this.#timer.destroy();
		this.#sortingTelemetryBinding?.dispose();
		this.#renderingTelemetryBinding?.dispose();
	}
}

type rendererConstructor<T extends Renderer> = new(device: GPUDevice, common: CommonRendererData, ...args: any[]) => T;
export default class RendererFactory {
	#commonData: CommonRendererData;

	constructor(device: GPUDevice, canvasContext: GPUCanvasContext) {
		this.#commonData = new CommonRendererData(device, canvasContext);
	}

	createRenderer<T extends Renderer>(device: GPUDevice, type: rendererConstructor<T>, ...args: any[]): T {
		return new type(device, this.#commonData, ...args);
	}

	destroy() {
		this.#commonData.destroy();
	}
}

class CommonRendererData {
	cameraUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	controlsUniformsLayoutEntry: GPUBindGroupLayoutEntry;
	gaussiansLayoutEntry: GPUBindGroupLayoutEntry;
	cameraUniforms: StructuredView;
	controlsUniforms: StructuredView;
	cameraUniformsBuffer: GPUBuffer;
	controlsUniformsBuffer: GPUBuffer;
	primaryRenderBindGroupLayout: GPUBindGroupLayout;

	canvasContext: GPUCanvasContext

	constructor(device: GPUDevice, canvasContext: GPUCanvasContext) {
		this.canvasContext= canvasContext;

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
		const shaderData = makeShaderDataDefinitions(rasterCode);
		this.cameraUniforms = makeStructuredView(shaderData.structs.Camera);
		this.controlsUniforms = makeStructuredView(shaderData.structs.RenderControls);
		this.controlsUniforms.set({ maxSH: 3, scaleMod: 1 });

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

		this.primaryRenderBindGroupLayout = device.createBindGroupLayout({
			label: "render bindgroup layout",
			entries: [
				this.cameraUniformsLayoutEntry,
				this.gaussiansLayoutEntry,
				this.controlsUniformsLayoutEntry
			]
		});
	}

	destroy() {
		this.cameraUniformsBuffer.destroy();
		this.controlsUniformsBuffer.destroy();
	}
}