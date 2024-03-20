import { CameraData } from "./loadCameras";
import { GPUSplats } from "./loadGaussians";
import CommonRendererData from "./renderers/common";

export default class Scene {
	splats: GPUSplats;
	cameras?: Array<CameraData>;

	renderBindGroup: GPUBindGroup;

	constructor(device: GPUDevice, rendererData: CommonRendererData, splats: GPUSplats) {
		this.splats = splats;

		this.renderBindGroup = device.createBindGroup({
			label: "render bindgroup",
			layout: rendererData.primaryRenderBindGroupLayout,
			entries: [{
					binding: rendererData.cameraUniformsLayoutEntry.binding,
					resource: { buffer: rendererData.cameraUniformsBuffer }
				}, {
					binding: rendererData.gaussiansLayoutEntry.binding,
					resource: { buffer: splats.buffer }
				}, {
					binding: rendererData.controlsUniformsLayoutEntry.binding,
					resource: { buffer: rendererData.controlsUniformsBuffer }
				}
			]
		});
	}

	destroy() {
		this.splats.buffer.destroy();
	}
}