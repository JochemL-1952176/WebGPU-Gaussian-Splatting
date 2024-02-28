import { CameraData } from "./loadCameras";
import { GPUSplats } from "./loadGaussians";
import Renderer from "./renderer";

export default class Scene {
	splats: GPUSplats;
	cameras?: Array<CameraData>;

	renderBindGroup: GPUBindGroup;

	constructor(device: GPUDevice, renderer: Renderer, splats: GPUSplats) {
		this.splats = splats;

		this.renderBindGroup = device.createBindGroup({
			label: "render bindgroup",
			layout: renderer.primaryRenderBindGroupLayout,
			entries: [{
					binding: renderer.cameraUniformsLayoutEntry.binding,
					resource: { buffer: renderer.cameraUniformsBuffer }
				}, {
					binding: renderer.gaussiansLayoutEntry.binding,
					resource: { buffer: splats.buffer }
				}, {
					binding: renderer.controlsUniformsLayoutEntry.binding,
					resource: { buffer: renderer.controlsUniformsBuffer }
				}
			]
		});

		renderer.finalize(device, this);
	}

	destroy() {
		this.splats.buffer.destroy();
	}
}