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
	const ret = files.map((value: string) => ({
		dir: value,
		key: parseInt(value.slice(value.lastIndexOf('/') + 1, value.lastIndexOf('.')))
	}));
	
	return ret.sort(({key: keyA}, {key: keyB}) => keyA - keyB);
}

const thresholdMaps = {
	bayer: importThresholdMaps(Object.keys(import.meta.glob('@assets/bayer/*.png', { query: '?url', import: "default" }))),
	blueNoise: importThresholdMaps(Object.keys(import.meta.glob('@assets/blueNoise/*.png', { query: '?url', import: "default" })))
};

type thresholdMapType = keyof typeof thresholdMaps;

const selectedThresholdMap = {
	type: "bayer" as thresholdMapType,
	size: 0,
};

function loadBitmap(type: thresholdMapType, size: number) {
	selectedThresholdMap.type = type;
	selectedThresholdMap.size = size;

	return fetch(thresholdMaps[type][size].dir)
	.then(async response => response.blob())
	.then(blob => createImageBitmap(blob));
}
let thresholdImage = await loadBitmap(selectedThresholdMap.type, selectedThresholdMap.size);

const medianFilterSettings = {
	enabled: false,
	size: 3,
}

export class StochasticRenderer extends Renderer {
	#rasterShader: GPUShaderModule;
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
	
	#thresholdMapTypeSelectionBinding?: RadioGridApi<thresholdMapType>;
	#bayerMapSelectionBinding?: RadioGridApi<number>;
	#blueNoiseMapSelectionBinding?: RadioGridApi<number>;
	#medianFilterToggleBinding?: BindingApi<unknown, boolean>;
	#medianFilterSizeBinding?: BindingApi<unknown, number>;
	#renderingTelemetryBinding?: BindingApi<unknown, number>;
	#filterTelemetryBinding?: BindingApi<unknown, number>;
	
	constructor(device: GPUDevice, common: CommonRendererData) {
		super(device, common);
		this.#renderFormat = this.common.canvasContext.getCurrentTexture().format;

		this.#timer = new GPUTimer(device, 2);

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode + stochasticRasterizeShaderCode;
		this.#rasterShader = device.createShaderModule({ code: rasterCode });

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
		
		this.#filterRenderpassDescriptor = {
			colorAttachments: [{
				view: unsetRenderTarget,
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store"
			}]
		};

		this.setSize(device, this.common.canvasContext.canvas.width, this.common.canvasContext.canvas.height);
		this.#buildFilterRenderPipeline(device);
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
		const thresholdMapTypes = Object.keys(thresholdMaps) as Array<thresholdMapType>;

		this.#thresholdMapTypeSelectionBinding = root.addBlade({
			view: 'radiogrid',
			groupName: 'thresholdMap',
			label: "Threshold map type",
			size: [thresholdMapTypes.length, 1],
			cells: (x: number, _y: number) => {
				const value = thresholdMapTypes[x];
				const title = value.split(/(?=[A-Z])/)
					.map(w => w.charAt(0).toUpperCase() + w.slice(1))
					.join(' ')

				return { value, title }
			},
			value: selectedThresholdMap.type
		});

		this.#bayerMapSelectionBinding = root.addBlade({
			view: 'radiogrid',
			groupName: 'BayerThresholdMap',
			label: 'Bayer',
			size: [thresholdMaps.bayer.length, 1],
			cells: (x: number, _y: number) => ({
				value: x,
				title: thresholdMaps.bayer[x].key
			}),
			value: selectedThresholdMap.size
		}) as RadioGridApi<number>;

		this.#blueNoiseMapSelectionBinding = root.addBlade({
			view: 'radiogrid',
			groupName: 'blueNoisethresholdMap',
			label: 'Blue noise',
			size: [thresholdMaps.blueNoise.length, 1],
			cells: (x: number, _y: number) => ({
				value: x,
				title: thresholdMaps.blueNoise[x].key
			}),
			value: selectedThresholdMap.size
		}) as RadioGridApi<number>;

		this.#bayerMapSelectionBinding.hidden = selectedThresholdMap.type !== "bayer";
		this.#blueNoiseMapSelectionBinding.hidden = selectedThresholdMap.type !== "blueNoise";

		this.#thresholdMapTypeSelectionBinding!.on('change', (e) => {
			selectedThresholdMap.type = e.value;

			if (selectedThresholdMap.type === "bayer") {
				this.#bayerMapSelectionBinding!.hidden = false;
				this.#blueNoiseMapSelectionBinding!.hidden = true;
				selectedThresholdMap.size = this.#bayerMapSelectionBinding!.value.rawValue;
			}
			
			if (selectedThresholdMap.type === "blueNoise") {
				this.#bayerMapSelectionBinding!.hidden = true;
				this.#blueNoiseMapSelectionBinding!.hidden = false;
				selectedThresholdMap.size = this.#blueNoiseMapSelectionBinding!.value.rawValue;
			}
			loadBitmap(selectedThresholdMap.type, selectedThresholdMap.size).then(image => this.#setThresholdMap(device, image));
		})

		this.#bayerMapSelectionBinding.on('change', (e) => {
			loadBitmap("bayer", e.value).then(image => this.#setThresholdMap(device, image));
		});

		this.#blueNoiseMapSelectionBinding.on('change', (e) => {
			loadBitmap("blueNoise", e.value).then(image => this.#setThresholdMap(device, image));
		});

		this.#medianFilterToggleBinding = root.addBinding(medianFilterSettings, 'enabled', { label: "Median filter" });
		this.#medianFilterSizeBinding = root.addBinding(medianFilterSettings, 'size', {
			label: "Filter size",
			hidden: !medianFilterSettings.enabled,
			options: {
				"3x3": 3,
				"5x5": 5,
				"7x7": 7,
				"9x9": 9,
			}
		});
		
		this.#medianFilterToggleBinding.on('change', (e) => {
				this.#filterTelemetryBinding!.hidden = !e.value
				this.#medianFilterSizeBinding!.hidden = !e.value
		});

		this.#medianFilterSizeBinding.on('change', () => {
			this.#buildFilterRenderPipeline(device);
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
			hidden: !medianFilterSettings.enabled,
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

	#buildFilterRenderPipeline(device: GPUDevice) {
		const CorrectfilterShaderCode = filterShaderCode.replaceAll('FILTERSIZE', medianFilterSettings.size.toString());
		const filterShader = device.createShaderModule({ code: CorrectfilterShaderCode });
		const filterPipelineLayout = device.createPipelineLayout({
			label: "Filter pipeline layout",
			bindGroupLayouts: [this.#filterBindGroupLayout]
		});

		this.#filterRenderPipeline = device.createRenderPipeline({
			label: "Filter pipeline",
			primitive: { topology: "triangle-list" },
			layout: filterPipelineLayout,
			vertex: { module: filterShader, entryPoint: "vs" },
			fragment: {
				module: filterShader,
				entryPoint: "fs",
				targets: [{ format: this.#renderFormat }]
			}
		});

		this.#buildFilterRenderBundle(device);
	}

	#buildFilterRenderBundle(device: GPUDevice) {
		if (!this.#filterRenderPipeline) return;

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
		thresholdImage = image;
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
		this.#thresholdMapTypeSelectionBinding?.dispose();
		this.#bayerMapSelectionBinding?.dispose();
		this.#blueNoiseMapSelectionBinding?.dispose();
		this.#medianFilterToggleBinding?.dispose();
		this.#medianFilterSizeBinding?.dispose();
		this.#renderingTelemetryBinding?.dispose();
		this.#filterTelemetryBinding?.dispose();
	}
}