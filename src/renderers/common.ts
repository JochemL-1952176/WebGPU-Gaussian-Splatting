import { StructuredView, makeShaderDataDefinitions, makeStructuredView } from "webgpu-utils";
import sharedShaderCode from '../shaders/shared.wgsl?raw';
import sharedRasterizeShaderCode from '../shaders/sharedRasterize.wgsl?raw';

export default class CommonRendererData {
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

		const rasterCode = sharedShaderCode + sharedRasterizeShaderCode;
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