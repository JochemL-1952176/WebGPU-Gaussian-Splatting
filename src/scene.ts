import { CameraData } from "./loadCameras";
import { GPUSplats } from "./loadGaussians";
import { Renderer } from "./renderer";

export default class Scene {
	splats: GPUSplats;
	cameras?: Array<CameraData>;

	renderBindGroup: GPUBindGroup;

	constructor(device: GPUDevice, renderer: Renderer, splats: GPUSplats) {
		this.splats = splats;

		this.renderBindGroup = device.createBindGroup({
			label: "render bindgroup",
			layout: renderer.common.primaryRenderBindGroupLayout,
			entries: [{
					binding: renderer.common.cameraUniformsLayoutEntry.binding,
					resource: { buffer: renderer.common.cameraUniformsBuffer }
				}, {
					binding: renderer.common.gaussiansLayoutEntry.binding,
					resource: { buffer: splats.buffer }
				}, {
					binding: renderer.common.controlsUniformsLayoutEntry.binding,
					resource: { buffer: renderer.common.controlsUniformsBuffer }
				}
			]
		});

		renderer.finalize(device, this);
	}

	destroy() {
		this.splats.buffer.destroy();
	}
}