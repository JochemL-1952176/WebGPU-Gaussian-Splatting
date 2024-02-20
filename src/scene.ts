import Renderer from "./renderer";

export type GPUSplats = {
	count: number,
	buffer: GPUBuffer
};

export default class Scene {
	#renderer: Renderer;
	#splats?: GPUSplats;

	renderBundle!: GPURenderBundle;
	#renderBindGroup!: GPUBindGroup;

	#cameraUniformsBindgroupEntry: GPUBindGroupEntry;
	#controlsUniformsBindgroupEntry: GPUBindGroupEntry;
	#gaussiansBindgroupEntry: GPUBindGroupEntry;
	#renderBindGroupDescriptor: GPUBindGroupDescriptor;

	constructor(device: GPUDevice, renderer: Renderer, splats: GPUSplats) {
		this.#renderer = renderer;

		this.#cameraUniformsBindgroupEntry = {
			binding: renderer.cameraUniformsLayoutEntry.binding,
			resource: { buffer: renderer.cameraUniformsBuffer }
		};

		this.#controlsUniformsBindgroupEntry = {
			binding: renderer.controlsUniformsLayoutEntry.binding,
			resource: { buffer: renderer.controlsUniformsBuffer }
		};

		this.#gaussiansBindgroupEntry = {
			binding: renderer.gaussiansLayoutEntry.binding,
			resource: { buffer: splats.buffer }
		};

		this.#renderBindGroupDescriptor = {
			label: "render bindgroup",
			layout: renderer.renderBindGroupLayout,
			entries: [
				this.#cameraUniformsBindgroupEntry,
				this.#controlsUniformsBindgroupEntry,
				this.#gaussiansBindgroupEntry
			]
		};

		this.setSplats(device, splats);
	}

	#buildRenderBundle(device: GPUDevice) {
		const bundleEncoder = device.createRenderBundleEncoder({
			colorFormats: [this.#renderer.canvasFormat],
			depthStencilFormat: this.#renderer.renderPipelineDescriptor.depthStencil!.format
		});
		
		bundleEncoder.setPipeline(this.#renderer.renderPipeline);
		bundleEncoder.setBindGroup(0, this.#renderBindGroup);
		
		bundleEncoder.draw(4, this.#splats!.count);
		return bundleEncoder.finish();
	}

	setSplats(device: GPUDevice, splats: GPUSplats) {
		this.#splats?.buffer.destroy();
		this.#splats = splats;

		(this.#gaussiansBindgroupEntry.resource as GPUBufferBinding).buffer = splats.buffer;
		this.#renderBindGroupDescriptor.entries = [
			this.#cameraUniformsBindgroupEntry,
			this.#controlsUniformsBindgroupEntry,
			this.#gaussiansBindgroupEntry
		];
		this.#renderBindGroup = device.createBindGroup(this.#renderBindGroupDescriptor);
		this.renderBundle = this.#buildRenderBundle(device);
	}
}