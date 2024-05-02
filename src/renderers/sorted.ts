import { BindingApi, FolderApi } from "@tweakpane/core";
import GPUTimer from "../GPUTimer";
import SplatSorter from "../sorter";
import { Renderer } from "./renderer";
import CommonRendererData, { unsetRenderTarget } from "./common";
import Scene from "../scene";

import sharedShaderCode from '@shaders/shared.wgsl?raw';
import sharedRasterizeShaderCode from '@shaders/sharedRasterize.wgsl?raw';
import sortedRasterizeShaderCode from '@shaders/sortedRasterize.wgsl?raw';
import { Camera } from "../cameraControls";
import { Pane } from "tweakpane";

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
		super(device, common);

		this.#timer = new GPUTimer(device, 6);

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode + sortedRasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });
		
		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: unsetRenderTarget,
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
	controlPanes(_root: FolderApi | Pane, _device: GPUDevice): void {};

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