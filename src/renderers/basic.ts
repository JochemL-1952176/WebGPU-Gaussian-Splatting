import { BindingApi, FolderApi } from "@tweakpane/core";
import GPUTimer from "../GPUTimer";
import { Renderer } from "./renderer";
import CommonRendererData from "./common";

import sharedShaderCode from '../shaders/shared.wgsl?raw';
import sharedRasterizeShaderCode from '../shaders/sharedRasterize.wgsl?raw';
import basicRasterizeShaderCode from '../shaders/basicRasterize.wgsl?raw';
import Scene from "../scene";
import { Camera } from "../cameraControls";
import { Pane } from "tweakpane";

export class BasicRenderer extends Renderer {
	#rasterShader: GPUShaderModule;
	#renderPassDescriptor: GPURenderPassDescriptor;
	#depthFormat: GPUTextureFormat = "depth32float";
	#depthBuffer: GPUTexture;
	#thresholdControlsUniform: Float32Array;
	#thresholdControlsBuffer: GPUBuffer;
	
	#timer: GPUTimer;
	#renderBundle?: GPURenderBundle;

	#telemetry = {
		rendering: 0
	};

	#thresholdControlsBinding?: BindingApi<unknown, number>
	#renderingTelemetryBinding?: BindingApi<unknown, number>;
	
	constructor(device: GPUDevice, common: CommonRendererData) {
		super(device, common);

		this.#timer = new GPUTimer(device, 1);

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode + basicRasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });
		
		this.#depthBuffer = device.createTexture({
			format: this.#depthFormat,
			size: [common.canvasContext.canvas.width, common.canvasContext.canvas.height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});

		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: this.common.canvasContext.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}],
			depthStencilAttachment: {
				view: this.#depthBuffer.createView(),
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store"
			}
		};

		this.#thresholdControlsUniform = new Float32Array(1);
		this.#thresholdControlsUniform[0] = 0.2;

		this.#thresholdControlsBuffer = device.createBuffer({
			size: this.#thresholdControlsUniform.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		
		device.queue.writeBuffer(this.#thresholdControlsBuffer, 0, this.#thresholdControlsUniform);
		this.setSize(device, this.common.canvasContext.canvas.width, this.common.canvasContext.canvas.height);
	}

	finalize(device: GPUDevice, scene: Scene) {
		
		const secondaryRenderBindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" }
			}]
		});

		const secondaryRenderBindGroup = device.createBindGroup({
			layout: secondaryRenderBindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: this.#thresholdControlsBuffer }
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
				targets: [{ format: this.common.canvasContext.getCurrentTexture().format }]
			},
			depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: this.#depthFormat
			}
		});

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.common.canvasContext.getCurrentTexture().format],
			depthStencilFormat: this.#depthFormat
		});

		renderPassEncoder.setPipeline(renderPipeline);
		renderPassEncoder.setBindGroup(0, scene.renderBindGroup);
		renderPassEncoder.setBindGroup(1, secondaryRenderBindGroup);
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

		this.#renderPassDescriptor.colorAttachments[0].view = this.common.canvasContext.getCurrentTexture().createView();

		const renderPass = this.#timer.beginRenderPass(encoder, this.#renderPassDescriptor, 0);
		renderPass.executeBundles([this.#renderBundle!]);
		renderPass.end();

		device.queue.submit([encoder.finish()]);
		this.#getTimings();
	}

	setSize(device: GPUDevice, width: number, height: number) {
		this.#depthBuffer.destroy();
		this.#depthBuffer = device.createTexture({
			format: this.#depthFormat,
			size: [width, height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});

		this.#renderPassDescriptor.depthStencilAttachment!.view = this.#depthBuffer.createView();
	}

	controlPanes(root: FolderApi | Pane, device: GPUDevice): void {
		this.#thresholdControlsBinding = root.addBinding((this.#thresholdControlsUniform as any), '0', {
			label: "Alpha threshold",
			min: 0, max: 1, step: 0.01,
		});

		this.#thresholdControlsBinding.on('change', () => device.queue.writeBuffer(this.#thresholdControlsBuffer, 0, this.#thresholdControlsUniform));
	};

	telemetryPanes(root: FolderApi | Pane, interval: number = 50) {
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
				this.#telemetry.rendering = results[0];
			}
		});
	}

	destroy() {
		this.#timer.destroy();
		this.#thresholdControlsBinding?.dispose();
		this.#renderingTelemetryBinding?.dispose();
	}
}