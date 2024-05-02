import { BindingApi, FolderApi } from "@tweakpane/core";
import GPUTimer from "../GPUTimer";
import { Renderer } from "./renderer";
import CommonRendererData, { unsetRenderTarget } from "./common";

import sharedShaderCode from '@shaders/shared.wgsl?raw';
import sharedRasterizeShaderCode from '@shaders/sharedRasterize.wgsl?raw';
import RasterizeShaderCode from '@shaders/weightedBlended/rasterize.wgsl?raw';
import BlendShaderCode from '@shaders/weightedBlended/blend.wgsl?raw';
import Scene from "../scene";
import { Camera } from "../cameraControls";
import { Pane } from "tweakpane";

// Webgpu does not yet support blending rgba32float textures, so we have to settle for 16 bit textures,
// which likely does not meaningfully change the quality or render time of the produced frames
// https://github.com/gpuweb/gpuweb/issues/3556
const accumulateFormat: GPUTextureFormat = "rgba16float"
const revealageFormat: GPUTextureFormat = "rgba16float"

export class WeightedBlendedRenderer extends Renderer {
	#rasterShader: GPUShaderModule;
	#blendShader: GPUShaderModule;
	#renderPassDescriptor: GPURenderPassDescriptor;
	#blendBindGroupLayout: GPUBindGroupLayout;
	#blendBindGroupDescriptor: GPUBindGroupDescriptor;
	#blendPipeline: GPURenderPipeline;
	#blendRenderPassDescriptor: GPURenderPassDescriptor;
	
	#accumulateTexture?: GPUTexture;
	#revealageTexture?: GPUTexture;

	#timer: GPUTimer;
	#renderBundle?: GPURenderBundle;
	#blendRenderbundle?: GPURenderBundle;

	#telemetry = {
		rendering: 0,
		blending: 0,
	};

	#renderingTelemetryBinding?: BindingApi<unknown, number>;
	#blendingTelemetryBinding?: BindingApi<unknown, number>;
	
	constructor(device: GPUDevice, common: CommonRendererData) {
		super(device, common);

		this.#timer = new GPUTimer(device, 2);

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode + RasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });
		this.#blendShader = device.createShaderModule({ code: BlendShaderCode });
		
		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: unsetRenderTarget,
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}, {
				view: unsetRenderTarget,
				loadOp: "clear",
				clearValue: [1, 1, 1, 1],
				storeOp: "store"
			}]
		};

		this.#blendBindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "unfilterable-float" }
			}, {
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "unfilterable-float" }
			}]
		});

		this.#blendBindGroupDescriptor = {
			label: "Blend pass bind group descriptor",
			layout: this.#blendBindGroupLayout,
			entries: [{
				binding: 0,
				resource: unsetRenderTarget
			}, {
				binding: 1,
				resource: unsetRenderTarget
			}]
		};

		const blendPipelineLayout = device.createPipelineLayout({
			label: "blend pipeline layout",
			bindGroupLayouts: [this.#blendBindGroupLayout]
		});

		const blend: GPUBlendComponent = {
			srcFactor: "one-minus-src-alpha",
			dstFactor: "src-alpha"
		}
		
		this.#blendPipeline = device.createRenderPipeline({
			label: "blend pipeline",
			primitive: { topology: 'triangle-list' },
			layout: blendPipelineLayout,
			vertex: { module: this.#blendShader, entryPoint: "vs" },
			fragment: {
				module: this.#blendShader,
				entryPoint: "fs",
				targets: [{
					format: this.common.canvasContext.getCurrentTexture().format,
					blend: { color: blend, alpha: blend }
				}]
			}
		});

		this.#blendRenderPassDescriptor = {
			colorAttachments: [{
				view: unsetRenderTarget,
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store"
			}]
		};
		
		this.setSize(device, this.common.canvasContext.canvas.width, this.common.canvasContext.canvas.height);
	}

	finalize(device: GPUDevice, scene: Scene) {
		const renderPipelineLayout = device.createPipelineLayout({
			label: "render pipeline layout",
			bindGroupLayouts: [this.common.primaryRenderBindGroupLayout]
		});
		
		const accumulateBlend: GPUBlendComponent = {
			srcFactor: "one",
			dstFactor: "one"
		}

		const revealageBlend: GPUBlendComponent = {
			srcFactor: "zero",
			dstFactor: "one-minus-src-alpha"
		}

		const renderPipeline = device.createRenderPipeline({
			label: "render pipeline",
			primitive: { topology: 'triangle-strip' },
			layout: renderPipelineLayout,
			vertex: { module: this.#rasterShader, entryPoint: "vs" },
			fragment: {
				module: this.#rasterShader,
				entryPoint: "fs",
				targets: [{
					format: accumulateFormat,
					blend: { color: accumulateBlend, alpha: accumulateBlend }
				}, {
					format: revealageFormat,
					blend: { color: revealageBlend, alpha: revealageBlend }
				}]
			}
		});

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [accumulateFormat, revealageFormat]
		});

		renderPassEncoder.setPipeline(renderPipeline);
		renderPassEncoder.setBindGroup(0, scene.renderBindGroup);
		renderPassEncoder.draw(4, scene.splats.count);

		this.#renderBundle = renderPassEncoder.finish();
	}

	renderFrame(device: GPUDevice, _scene: Scene, camera: Camera) {
		console.assert(this.#renderBundle !== undefined, "Call finalize before rendering");
		const encoder = device.createCommandEncoder();
		
		if (camera.hasChanged) {		
			device.queue.writeBuffer(this.common.cameraUniformsBuffer, 0, camera.uniforms.arrayBuffer);
			camera.hasChanged = false;
		}

		const renderPass = this.#timer.beginRenderPass(encoder, this.#renderPassDescriptor, 0);
		renderPass.executeBundles([this.#renderBundle!]);
		renderPass.end();

		this.#blendRenderPassDescriptor.colorAttachments[0].view = this.common.canvasContext.getCurrentTexture().createView();
		const blendPass = this.#timer.beginRenderPass(encoder, this.#blendRenderPassDescriptor, 1);
		blendPass.executeBundles([this.#blendRenderbundle!]);
		blendPass.end();

		device.queue.submit([encoder.finish()]);
		this.#getTimings();
	}

	setSize(device: GPUDevice, width: number, height: number) {
		this.#accumulateTexture?.destroy();
		this.#accumulateTexture = device.createTexture({
			format: accumulateFormat,
			size: [width, height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});

		this.#revealageTexture?.destroy();
		this.#revealageTexture = device.createTexture({
			format: revealageFormat,
			size: [width, height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});

		this.#renderPassDescriptor.colorAttachments[0]!.view = this.#accumulateTexture.createView();
		this.#blendBindGroupDescriptor.entries[0].resource = this.#accumulateTexture.createView();
		this.#renderPassDescriptor.colorAttachments[1]!.view = this.#revealageTexture.createView();
		this.#blendBindGroupDescriptor.entries[1].resource = this.#revealageTexture.createView();

		const blendBindGroup = device.createBindGroup(this.#blendBindGroupDescriptor);

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.common.canvasContext.getCurrentTexture().format]
		});

		renderPassEncoder.setPipeline(this.#blendPipeline);
		renderPassEncoder.setBindGroup(0, blendBindGroup);
		renderPassEncoder.draw(3);

		this.#blendRenderbundle = renderPassEncoder.finish();
	}

	controlPanes(_root: FolderApi | Pane, _device: GPUDevice): void {};

	telemetryPanes(root: FolderApi | Pane, interval: number = 50) {
		this.#renderingTelemetryBinding = root.addBinding(this.#telemetry, 'rendering', {
			readonly: true,
			view: 'graph',
			label: 'Render time',
			format: (v: number) => `${v.toFixed(2)}μs`,
			min: 0, max: 100_000 / 3,
			interval: interval
		});

		this.#blendingTelemetryBinding = root.addBinding(this.#telemetry, 'blending', {
			readonly: true,
			view: 'graph',
			label: 'Blend time',
			format: (v: number) => `${v.toFixed(2)}μs`,
			min: 0, max: 100_000 / 3,
			interval: interval
		});
	}

	#getTimings() {
		this.#timer.getResults().then((results?: Array<number>) => {
			if (results) {
				this.#telemetry.rendering = results[0];
				this.#telemetry.blending = results[1];
			}
		});
	}

	destroy() {
		this.#timer.destroy();
		this.#renderingTelemetryBinding?.dispose();
		this.#blendingTelemetryBinding?.dispose();
	}
}