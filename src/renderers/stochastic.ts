import { BindingApi, FolderApi } from "@tweakpane/core";
import GPUTimer from "../GPUTimer";
import { Renderer } from "./renderer";
import CommonRendererData, { unsetRenderTarget } from "./common";
import sharedShaderCode from '@shaders/shared.wgsl?raw';
import sharedRasterizeShaderCode from '@shaders/sharedRasterize.wgsl?raw';
import stochasticRasterizeShaderCode from '@shaders/stochasticRasterize.wgsl?raw';
import filterShaderCode from '@shaders/medianFilter.wgsl?raw';
import Scene from "../scene";
import { Camera } from "../cameraControls";
import { Pane } from "tweakpane";
import { RadioGridApi } from "@tweakpane/plugin-essentials";
import { createTextureFromSource } from "webgpu-utils";

function importThresholdMaps(files: string[]) {
	const ret =files.map((value: string) => ({
		dir: value,
		key: parseInt(value.slice(value.lastIndexOf('/') + 1, value.lastIndexOf('.')))
	}));
	
	return ret.sort(({key: keyA}, {key: keyB}) => keyA - keyB);
}

const thresholdMaps = [
	importThresholdMaps(Object.keys(import.meta.glob('@assets/bayer/*.png', { query: '?url', import: "default" }))),
	importThresholdMaps(Object.keys(import.meta.glob('@assets/blueNoise/*.png', { query: '?url', import: "default" })))
];

const selectedThresholdMap = {
	type: 0,
	size: 1,
};

function loadSelectedAsBitmap() {
	return fetch(thresholdMaps[selectedThresholdMap.type][selectedThresholdMap.size].dir)
	.then(async response => response.blob())
	.then(blob => createImageBitmap(blob));
}
const thresholdImage = await loadSelectedAsBitmap();

const medianFilterSettings = {
	enabled: false
}

export class StochasticRenderer extends Renderer {
	#rasterShader: GPUShaderModule;
	#filterShader: GPUShaderModule;
	#renderPassDescriptor: GPURenderPassDescriptor;
	#depthFormat: GPUTextureFormat = "depth32float";
	#renderFormat: GPUTextureFormat;
	#depthBuffer?: GPUTexture;
	#intermediateRenderResult?: GPUTexture;

	#filterBindGroupLayout: GPUBindGroupLayout;
	#filterBindGroupDescriptor: GPUBindGroupDescriptor;
	#filterRenderBundle?: GPURenderBundle;
	#filterRenderpassDescriptor: GPURenderPassDescriptor;
	#filterRenderPipeline?: GPURenderPipeline;

	#thresholdTextureBindGrouplayout: GPUBindGroupLayout;
	#thresholdTextureBindGroupDescriptor: GPUBindGroupDescriptor;
	#thresholdTexture: GPUTexture;
	#thresholdTextureChanged = false;
	
	#timer: GPUTimer;
	#renderBundle?: GPURenderBundle;
	#renderPipeline?: GPURenderPipeline;
	
	#telemetry = {
		rendering: 0,
		filter: 0
	};
	
	#medianFilterSettingsBinding?: BindingApi<unknown, boolean>;
	#bayerMapSelectionBinding?: RadioGridApi<number>;
	#blueNoiseMapSelectionBinding?: RadioGridApi<number>;
	#renderingTelemetryBinding?: BindingApi<unknown, number>;
	#filterTelemetryBinding?: BindingApi<unknown, number>;
	
	constructor(device: GPUDevice, common: CommonRendererData) {
		super(device, common);
		this.#renderFormat = this.common.canvasContext.getCurrentTexture().format;

		this.#timer = new GPUTimer(device, 2);

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode + stochasticRasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });
		this.#filterShader = device.createShaderModule({ code: filterShaderCode });

		this.#depthBuffer = device.createTexture({
			format: this.#depthFormat,
			size: [common.canvasContext.canvas.width, common.canvasContext.canvas.height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});

		this.#thresholdTexture = createTextureFromSource(device, thresholdImage, {
			usage: GPUTextureUsage.TEXTURE_BINDING
		});

		this.#thresholdTextureBindGrouplayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "unfilterable-float" }
			}]
		});

		this.#thresholdTextureBindGroupDescriptor = {
			label: "Threshold texture bind group descriptor",
			layout: this.#thresholdTextureBindGrouplayout,
			entries: [{
				binding: 0,
				resource: this.#thresholdTexture.createView()
			}]
		};

		this.#renderPassDescriptor = {
			colorAttachments: [{
				view: unsetRenderTarget,
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			}],
			depthStencilAttachment: {
				view: unsetRenderTarget,
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store"
			}
		};

		this.#filterBindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "unfilterable-float" }
			}]
		});

		this.#filterBindGroupDescriptor = {
			label: "Filter pass bind group descriptor",
			layout: this.#filterBindGroupLayout,
			entries: [{
				binding: 0,
				resource: unsetRenderTarget
			}]
		};

		const filterPipelineLayout = device.createPipelineLayout({
			label: "Filter pipeline layout",
			bindGroupLayouts: [this.#filterBindGroupLayout]
		});
		
		this.#filterRenderPipeline = device.createRenderPipeline({
			label: "Filter pipeline",
			primitive: { topology: "triangle-list" },
			layout: filterPipelineLayout,
			vertex: { module: this.#filterShader, entryPoint: "vs" },
			fragment: {
				module: this.#filterShader,
				entryPoint: "fs",
				targets: [{ format: this.#renderFormat }]
			}
		});

		this.#filterRenderpassDescriptor = {
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
			bindGroupLayouts: [this.common.primaryRenderBindGroupLayout, this.#thresholdTextureBindGrouplayout]
		});
		
		this.#renderPipeline = device.createRenderPipeline({
			label: "render pipeline",
			primitive: { topology: 'triangle-strip' },
			layout: renderPipelineLayout,
			vertex: { module: this.#rasterShader, entryPoint: "vs" },
			fragment: {
				module: this.#rasterShader,
				entryPoint: "fs",
				targets: [{ format: this.#renderFormat }]
			},
			depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: this.#depthFormat
			}
		});

		this.#buildRenderBundle(device, scene);
	}

	renderFrame(device: GPUDevice, scene: Scene, camera: Camera) {
		console.assert(this.#renderBundle !== undefined && this.#renderPipeline !== undefined, "Call finalize before rendering");
		if (this.#thresholdTextureChanged) {
			this.#thresholdTextureBindGroupDescriptor.entries[0].resource = this.#thresholdTexture.createView();
			this.#buildRenderBundle(device, scene);
			this.#thresholdTextureChanged = false;
		}

		const encoder = device.createCommandEncoder();
		
		if (camera.hasChanged) {		
			device.queue.writeBuffer(this.common.cameraUniformsBuffer, 0, camera.uniforms.arrayBuffer);
			camera.hasChanged = false;
		}

		this.#renderPassDescriptor.colorAttachments[0].view = medianFilterSettings.enabled ? 
			this.#intermediateRenderResult!.createView()
			: this.common.canvasContext.getCurrentTexture().createView();

			const renderPass = this.#timer.beginRenderPass(encoder, this.#renderPassDescriptor, 0);
			renderPass.executeBundles([this.#renderBundle!]);
			renderPass.end();

		if (medianFilterSettings.enabled) {
			this.#filterRenderpassDescriptor.colorAttachments[0].view = this.common.canvasContext.getCurrentTexture().createView();
			const filterPass = this.#timer.beginRenderPass(encoder, this.#filterRenderpassDescriptor, 1);
			filterPass.executeBundles([this.#filterRenderBundle!]);
			filterPass.end();
		}

		device.queue.submit([encoder.finish()]);
		this.#getTimings();
	}

	setSize(device: GPUDevice, width: number, height: number) {
		this.#depthBuffer?.destroy();
		this.#intermediateRenderResult?.destroy();

		this.#depthBuffer = device.createTexture({
			format: this.#depthFormat,
			size: [width, height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});

		this.#intermediateRenderResult = device.createTexture({
			format: this.#renderFormat,
			size: [width, height],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});

		this.#renderPassDescriptor.depthStencilAttachment!.view = this.#depthBuffer.createView();
		this.#renderPassDescriptor.colorAttachments[0].view = this.#intermediateRenderResult.createView();
		this.#filterBindGroupDescriptor.entries[0].resource = this.#intermediateRenderResult.createView();
		this.#buildFilterRenderBundle(device);
	}

	controlPanes(root: FolderApi | Pane, device: GPUDevice): void {
		this.#bayerMapSelectionBinding = root.addBlade({
			view: 'radiogrid',
			groupName: 'thresholdMap',
			label: 'Bayer',
			size: [thresholdMaps[0].length, 1],
			cells: (x: number, _y: number) => ({
				value: x,
				title: thresholdMaps[0][x].key
			}),
			value: selectedThresholdMap.type == 0 ? selectedThresholdMap.size : -1
		}) as RadioGridApi<number>;

		this.#blueNoiseMapSelectionBinding = root.addBlade({
			view: 'radiogrid',
			groupName: 'thresholdMap',
			label: 'Blue noise',
			size: [thresholdMaps[1].length, 1],
			cells: (x: number, _y: number) => ({
				value: x,
				title: thresholdMaps[1][x].key
			}),
			value: selectedThresholdMap.type == 1 ? selectedThresholdMap.size : -1,
		}) as RadioGridApi<number>;

		this.#medianFilterSettingsBinding = root.addBinding(medianFilterSettings, 'enabled', { label: "Median filter" })

		this.#medianFilterSettingsBinding.on('change', (e) => {
			if (this.#filterTelemetryBinding) {
				this.#filterTelemetryBinding.disabled = !e.value;
			}
		});

		this.#bayerMapSelectionBinding.on('change', (e) => {
			selectedThresholdMap.type = 0;
			selectedThresholdMap.size = e.value;
			loadSelectedAsBitmap().then(image => this.#setThresholdMap(device, image));
		});

		this.#blueNoiseMapSelectionBinding.on('change', (e) => {
			selectedThresholdMap.type = 1;
			selectedThresholdMap.size = e.value;
			loadSelectedAsBitmap().then(image => this.#setThresholdMap(device, image));
		});
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
		
		this.#filterTelemetryBinding = root.addBinding(this.#telemetry, 'filter', {
			readonly: true,
			disabled: !medianFilterSettings.enabled,
			view: 'graph',
			label: 'Median filter time',
			format: (v: number) => `${v.toFixed(2)}μs`,
			min: 0, max: 100_000 / 3,
			interval: interval
		});
	}

	#buildRenderBundle(device: GPUDevice, scene: Scene) {
		const secondaryRenderBindGroup = device.createBindGroup(this.#thresholdTextureBindGroupDescriptor);

		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.#renderFormat],
			depthStencilFormat: this.#depthFormat
		});

		renderPassEncoder.setPipeline(this.#renderPipeline!);
		renderPassEncoder.setBindGroup(0, scene.renderBindGroup);
		renderPassEncoder.setBindGroup(1, secondaryRenderBindGroup);
		renderPassEncoder.draw(4, scene.splats.count);

		this.#renderBundle = renderPassEncoder.finish();
	}

	#buildFilterRenderBundle(device: GPUDevice) {
		const filterBindGroup = device.createBindGroup(this.#filterBindGroupDescriptor);
		const renderPassEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.#renderFormat]
		});

		renderPassEncoder.setPipeline(this.#filterRenderPipeline!);
		renderPassEncoder.setBindGroup(0, filterBindGroup);
		renderPassEncoder.draw(3);
		
		this.#filterRenderBundle = renderPassEncoder.finish();
	}

	#setThresholdMap(device: GPUDevice, image: ImageBitmap) {
		this.#thresholdTexture.destroy();
		this.#thresholdTexture = createTextureFromSource(device, image, {
			usage: GPUTextureUsage.TEXTURE_BINDING
		});
		this.#thresholdTextureChanged = true;
	}

	#getTimings() {
		this.#timer.getResults().then((results?: Array<number>) => {
			if (results) {
				this.#telemetry.rendering = results[0];
				this.#telemetry.filter = results[1];
			}
		});
	}

	destroy() {
		this.#timer.destroy();
		this.#thresholdTexture.destroy();
		this.#bayerMapSelectionBinding?.dispose();
		this.#blueNoiseMapSelectionBinding?.dispose();
		this.#renderingTelemetryBinding?.dispose();
		this.#medianFilterSettingsBinding?.dispose();
	}
}