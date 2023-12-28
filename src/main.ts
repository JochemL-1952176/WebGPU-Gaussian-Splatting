import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { mat4, utils, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from './camera.js';
import readPLY from './readPLY.js';
import shaderCode from './shaders/gsplat.wgsl?raw'

if (!navigator.gpu) { throw new Error("WebGPU not supported in this browser"); }
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { throw new Error("No GPUAdapter found"); }

const device = await adapter.requestDevice({ requiredLimits: {
	maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
	maxBufferSize: 2 * Math.pow(1024, 3) // 2 GiB
}});

const canvas = document.querySelector("canvas")!;
const context = canvas.getContext("webgpu")!;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat,
	alphaMode: 'premultiplied'
});

const shaderModule = device.createShaderModule({ code: shaderCode });
const shaderData = makeShaderDataDefinitions(shaderCode);
const cameraUniforms = makeStructuredView(shaderData.structs.Camera);
const gaussianUniforms = makeStructuredView(shaderData.structs.GaussianOffsets);

const cameraUniformBuffer = device.createBuffer({
	label: "Camera uniforms buffer",
	size: cameraUniforms.arrayBuffer.byteLength,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const cameraUniformsBindGroupLayout = device.createBindGroupLayout({
	label: "Camera uniforms bind group layout",
	entries: [{
		binding: 0,
		visibility: GPUShaderStage.VERTEX,
		buffer: { type: "uniform" }
	}]
});

const uniformsBindGroup = device.createBindGroup({
	label: "Camera uniforms bind group",
	layout: cameraUniformsBindGroupLayout,
	entries: [{
		binding: 0,
		resource: { buffer: cameraUniformBuffer }
	}]
});

const GaussianBindGroupLayout = device.createBindGroupLayout({
	label: "Gaussians bind group layout",
	entries: [{
		binding: 0,
		visibility: GPUShaderStage.VERTEX,
		buffer: { type: "read-only-storage" }
	}, {
		binding: 1,
		visibility: GPUShaderStage.VERTEX,
		buffer: { type: "uniform" }
	}]
});

let numPoints = 0;
const gaussianBufferDescriptor: GPUBufferDescriptor = {
	label: "Gaussian data buffer",
	usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	size: Float32Array.BYTES_PER_ELEMENT
}
let gaussianBuffer = device.createBuffer(gaussianBufferDescriptor);

const gaussianUniformBuffer = device.createBuffer({
	label: "Gaussian offsets uniforms buffer",
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	size: gaussianUniforms.arrayBuffer.byteLength
});

const GaussianBindGroupDescriptor: GPUBindGroupDescriptor = {	
	label: "Gaussians bind group",
	layout: GaussianBindGroupLayout,
	entries: [{
		binding: 0,
		resource: { buffer: gaussianBuffer }
	}, {
		binding: 1,
		resource: { buffer: gaussianUniformBuffer }
	}]
};

let gaussianBindGroup = device.createBindGroup(GaussianBindGroupDescriptor);

const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
	label: "Pipeline",
	layout: device.createPipelineLayout({
		label: "Pipeline layout",
		bindGroupLayouts: [cameraUniformsBindGroupLayout, GaussianBindGroupLayout]
	}),
	primitive: { topology: 'triangle-strip' },
	vertex: {
		module: shaderModule,
		entryPoint: "vs"
	},
	fragment: {
		module: shaderModule,
		entryPoint: "fs",
		targets: [{
			format: canvasFormat,
			blend: {
				alpha: {
					srcFactor: 'one',
					dstFactor: 'one-minus-src-alpha'
				},
				color: {
					srcFactor: 'src-alpha',
					dstFactor: 'one-minus-src-alpha'
				}
			}
		}]
	},
	depthStencil: {
		depthWriteEnabled: true,
		format: "depth24plus",
		depthCompare: "less",
	}
}

const renderPipeline = device.createRenderPipeline(renderPipelineDescriptor);

var depthTexture = device.createTexture({
	size: [canvas.width, canvas.height],
	format: renderPipelineDescriptor.depthStencil!.format!,
	usage: GPUTextureUsage.RENDER_ATTACHMENT
});

const camera = new OrbitCamera(
	vec3.fromValues(0, 0, -2),
	vec3.fromValues(0, 0, 0),
	canvas,
);

camera.onChange = () => {
	cameraUniforms.set({ view: camera.getViewMatrix(), position: camera.position });

	device.queue.writeBuffer(
		cameraUniformBuffer,
		cameraUniforms.views.position.byteOffset,
		cameraUniforms.arrayBuffer,
		cameraUniforms.views.position.byteOffset,
		cameraUniforms.views.projection.byteOffset
	);
}

const frame = () => {
	camera.update();

	const encoder = device.createCommandEncoder();
	const renderPass = encoder.beginRenderPass({
		colorAttachments: [{
			view: context.getCurrentTexture().createView(),
			loadOp: "clear",
			clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1 },
			storeOp: "store",
		}],
		depthStencilAttachment: {
			view: depthTexture.createView(),
			depthClearValue: 1.0,
			depthLoadOp: "clear",
			depthStoreOp: "store"
		}
	});
	
	renderPass.setPipeline(renderPipeline);
	renderPass.setBindGroup(0, uniformsBindGroup);
	renderPass.setBindGroup(1, gaussianBindGroup);
	
	renderPass.draw(4, numPoints);
	
	renderPass.end();
	device.queue.submit([encoder.finish()]);
	requestAnimationFrame(frame);
}

const onResize = () => {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;

	console.log(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

	const projection = mat4.perspective(
		utils.degToRad(60),
		canvas.width / canvas.height,
		0.01, 1000
	);
	
	device.queue.writeBuffer(
		cameraUniformBuffer,
		cameraUniforms.views.projection.byteOffset,
		projection as Float32Array
	);

	depthTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		format: renderPipelineDescriptor.depthStencil!.format!,
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});
}

window.onresize = () => {
	console.log("Resizing");
	onResize();
};

onResize();
requestAnimationFrame(frame);

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
	document.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
});

document.ondrop = async (e) => {
	const files = e.dataTransfer?.files;
	if (files) {
		console.log(`Begin reading file: ${files[0].name}`);
		let begin = performance.now();
		const {header, data} = await readPLY(files[0]);
		let end = performance.now();
		console.log(header);
		numPoints = header.vertexCount;
		console.log(`Read ${numPoints} points in ${((end - begin) / 1000).toFixed(2)}s`);

		gaussianBuffer.destroy();
		gaussianBufferDescriptor.size = data.byteLength;
		gaussianBuffer = device.createBuffer(gaussianBufferDescriptor);

		console.log(`Uploading ${data.byteLength} bytes to GPU`);
		begin = performance.now();
		device.queue.writeBuffer(gaussianBuffer, 0, data);
		end = performance.now();
		console.log(`Uploading took ${((end - begin) / 1000).toFixed(2)}s`);
		
		GaussianBindGroupDescriptor.entries[0].resource.buffer = gaussianBuffer;
		gaussianBindGroup = device.createBindGroup(GaussianBindGroupDescriptor);

		gaussianUniforms.set({
			stride: header.stride,
			pos: header.properties["x"].offset,
			normal: header.properties["nx"].offset,
			opacity: header.properties["opacity"].offset,
			scale: header.properties["scale_0"].offset,
			rotation: header.properties["rot_0"].offset,
			sh: header.properties["f_dc_0"].offset,
		});

		device.queue.writeBuffer(gaussianUniformBuffer, 0, gaussianUniforms.arrayBuffer);
	}
}