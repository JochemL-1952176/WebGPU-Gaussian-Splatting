import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { mat4, utils, vec2, vec3 } from 'wgpu-matrix';
import { OrbitController } from './controls.js';
import debugGaussiansURL from './assets/default.ply?url' 
import shaderCode from './shaders/gsplat.wgsl?raw'
import { loadGaussianData } from './loadGaussians.js';

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
	alphaMode: "premultiplied"
});

const shaderModule = device.createShaderModule({ code: shaderCode });
const shaderData = makeShaderDataDefinitions(shaderCode);
const cameraUniforms = makeStructuredView(shaderData.structs.Camera);
const controlsUniforms = makeStructuredView(shaderData.structs.Controls);

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

const cameraUniformsBindGroup = device.createBindGroup({
	label: "Camera uniforms bind group",
	layout: cameraUniformsBindGroupLayout,
	entries: [{
		binding: 0,
		resource: { buffer: cameraUniformBuffer }
	}]
});

const controlsUniformBuffer = device.createBuffer({
	label: "Controls uniforms buffer",
	size: controlsUniforms.arrayBuffer.byteLength,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const controlsUniformsBindGrouplayout = device.createBindGroupLayout({
	label: "Controls uniforms bind group layout",
	entries: [{
		binding: 0,
		visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
		buffer: { type: "uniform" }
	}]
});

const controlsUniformsBindGroup = device.createBindGroup({
	label: "Controls uniforms bind group",
	layout: controlsUniformsBindGrouplayout,
	entries: [{
		binding: 0,
		resource: { buffer: controlsUniformBuffer }
	}]
});

const GaussianBindGroupLayout = device.createBindGroupLayout({
	label: "Gaussians bind group layout",
	entries: [{
		binding: 0,
		visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
		buffer: { type: "read-only-storage" }
	}]
});

let {count: numGaussians, gaussianBuffer} = await fetch(debugGaussiansURL)
	.then(async x => await loadGaussianData(await x.blob(), device))

const GaussianBindGroupDescriptor: GPUBindGroupDescriptor = {
	label: "Gaussians bind group",
	layout: GaussianBindGroupLayout,
	entries: [{
		binding: 0,
		resource: { buffer: gaussianBuffer }
	}]
};

let gaussianBindGroup = device.createBindGroup(GaussianBindGroupDescriptor);

const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
	label: "Pipeline",
	primitive: { topology: 'triangle-strip' },
	layout: device.createPipelineLayout({
		label: "Pipeline layout",
		bindGroupLayouts: [
cameraUniformsBindGroupLayout, 
			controlsUniformsBindGrouplayout,
						GaussianBindGroupLayout
		]
	}),
	vertex: { module: shaderModule, entryPoint: "vs" },
	fragment: {
		module: shaderModule,
		entryPoint: "fs",
		targets: [{
			format: canvasFormat,
			blend: {
				alpha: {
					srcFactor: "one",
					dstFactor: "one-minus-src-alpha"
				},
				color: {
					srcFactor: "src-alpha",
					dstFactor: "one-minus-src-alpha"
				}
			}
		}]
},
	depthStencil: {
		depthWriteEnabled: true,
		format: "depth16unorm",
		depthCompare: "less",
	}
};

const renderPipeline = await device.createRenderPipelineAsync(renderPipelineDescriptor);

const textureDescriptor = {
	size: [canvas.width, canvas.height],
	usage: GPUTextureUsage.RENDER_ATTACHMENT
};

let depthTexture = device.createTexture({
	format: renderPipelineDescriptor.depthStencil!.format!,
	...textureDescriptor
});

const renderPassDescriptor: GPURenderPassDescriptor = {
	colorAttachments: [{
		view: context.getCurrentTexture().createView(),
		loadOp: "clear",
		clearValue: { r: 0, g: 0, b: 0, a: 1 },
		storeOp: "store",
	}],
	depthStencilAttachment: {
		view: depthTexture.createView(),
		depthClearValue: 1.0,
		depthLoadOp: "clear",
		depthStoreOp: "store"
	}
};

const buildRenderBundle = () => {
	const bundleEncoder = device.createRenderBundleEncoder({
		colorFormats: [canvasFormat],
		depthStencilFormat: renderPipelineDescriptor.depthStencil!.format
	});
	
	bundleEncoder.setPipeline(renderPipeline);
	bundleEncoder.setBindGroup(0, cameraUniformsBindGroup);
	bundleEncoder.setBindGroup(1, controlsUniformsBindGroup);
	bundleEncoder.setBindGroup(2, gaussianBindGroup);
	
	bundleEncoder.draw(4, numGaussians);
	return bundleEncoder.finish();
}

let renderBundle = buildRenderBundle();

const controller = new OrbitController(
	vec3.fromValues(0, 0, -2),
	vec3.fromValues(0, 0, 0),
	canvas,
);

controller.onChange = () => {
	controller.getViewMatrix(cameraUniforms.views.view);
	cameraUniforms.set({ position: controller.position });
	const offset = cameraUniforms.views.position.byteOffset;
	const size = cameraUniforms.views.projection.byteOffset - offset;

	device.queue.writeBuffer(
		cameraUniformBuffer,
		offset,
		cameraUniforms.arrayBuffer,
		offset,
		size 
	);
};

let lastTime = performance.now();
const frame = () => {
	const time = performance.now();
	controller.update((time - lastTime) / 1000);
	lastTime = time

	const encoder = device.createCommandEncoder();
	renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
	const renderPass = encoder.beginRenderPass(renderPassDescriptor);

	renderPass.executeBundles([renderBundle])
	
	renderPass.end();
	device.queue.submit([encoder.finish()]);
	requestAnimationFrame(frame);
};

requestAnimationFrame(frame);

const onResize = () => {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;

	console.debug(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

	const aspect = canvas.width / canvas.height;
	const vFov = utils.degToRad(60);
	
	const htany = Math.tan(vFov / 2);
	const tan_fov = vec2.fromValues((htany / canvas.height) * canvas.width, htany);
	const focal = vec2.fromValues(canvas.width / (2 * tan_fov[0]), canvas.height / (2 * tan_fov[1]));
	
	cameraUniforms.set({tan_fov, focal});

	mat4.perspective(
		vFov,
		aspect,
		0.01, 1000,
		cameraUniforms.views.projection
	);

	const offset = cameraUniforms.views.projection.byteOffset;
	const size = cameraUniforms.arrayBuffer.byteLength - offset;

	device.queue.writeBuffer(
		cameraUniformBuffer,
		offset,
		cameraUniforms.arrayBuffer,
		offset,
		size 
	);

	textureDescriptor.size = [canvas.width, canvas.height];

	depthTexture.destroy();
	depthTexture = device.createTexture({format: renderPipelineDescriptor.depthStencil!.format!, ...textureDescriptor});
	renderPassDescriptor.depthStencilAttachment!.view = depthTexture.createView();
};

onResize();
window.onresize = onResize;

const plyInput = document.getElementById("PLYinput") as HTMLInputElement;
plyInput.onchange = async (_) => {
	if (plyInput.files && plyInput.files.length) {
		const fileType = plyInput.files[0].name.split('.').pop()!;
		if (fileType === "ply") {
			const {count, gaussianBuffer} = await loadGaussianData(plyInput.files[0], device);
			numGaussians = count;

			GaussianBindGroupDescriptor.entries[0].resource.buffer.destroy();
			GaussianBindGroupDescriptor.entries[0].resource.buffer = gaussianBuffer;
			gaussianBindGroup = device.createBindGroup(GaussianBindGroupDescriptor);

			renderBundle = buildRenderBundle();

		} else console.error(`Filetype ${fileType} is not supported`);
	}
};

//##############################//
//								//
//		UI-specific code		//
//								//
//##############################//

const SHSlider = document.getElementById("SHslider") as HTMLInputElement;
const SHDisplay = document.getElementById("SHDisplay") as HTMLOutputElement;
const SHUpdate = () => {
	SHDisplay.value = SHSlider.value;
	const val = parseInt(SHSlider.value);
	controlsUniforms.set({maxSH: val});
	device.queue.writeBuffer(
		controlsUniformBuffer,
		controlsUniforms.views.maxSH.byteOffset,
		controlsUniforms.views.maxSH
	);
}

SHUpdate();
SHSlider.oninput = SHUpdate;

const scaleSlider = document.getElementById("scaleSlider") as HTMLInputElement;
const scaleDisplay = document.getElementById("scaleDisplay") as HTMLOutputElement;
const scaleUpdate = () => {
	const val = parseFloat(scaleSlider.value)
	scaleDisplay.value = val.toFixed(2);
	controlsUniforms.set({scaleMod: val});
	device.queue.writeBuffer(
		controlsUniformBuffer,
		controlsUniforms.views.scaleMod.byteOffset,
		controlsUniforms.views.scaleMod
	);
}

scaleUpdate();
scaleSlider.oninput = scaleUpdate;