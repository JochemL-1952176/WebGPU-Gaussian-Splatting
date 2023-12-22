import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { mat4, utils, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from './camera.js';

if (!navigator.gpu) {
	throw new Error("WebGPU not supported in this browser");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
	throw new Error("No GPUAdapter found");
}

const device = await adapter.requestDevice();
const canvas = document.querySelector("canvas")!;
const context = canvas.getContext("webgpu")!;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat
});

const shaderCode = `
struct Camera {
	position: vec3f,
	projection: mat4x4f,
	view: mat4x4f
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexIn {
	@location(0) pos: vec2f,
	@location(1) uv: vec2f
};

struct FragmentIn {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f
};

@vertex fn vs(in: VertexIn) -> FragmentIn {
	return FragmentIn(
		cam.projection * cam.view * vec4f(in.pos, 0, 1),
		in.uv
	);
}

@fragment fn fs(in: FragmentIn) -> @location(0) vec4f {
	return vec4f(in.uv, 1.0 - length(in.uv), 1);
}`

const shaderModule = device.createShaderModule({
	label: "Shader",
	code: shaderCode
});

const shaderData = makeShaderDataDefinitions(shaderCode);
const uniforms = makeStructuredView(shaderData.structs.Camera);

const uniformBuffer = device.createBuffer({
	label: "Uniforms",
	size: uniforms.arrayBuffer.byteLength,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const bindGroupLayout = device.createBindGroupLayout({
	label: "Bind group layout",
	entries: [{
		binding: 0,
		visibility: GPUShaderStage.VERTEX,
		buffer: { type: "uniform" }
	}]
});

const bindGroup = device.createBindGroup({
	label: "Bind group",
	layout: bindGroupLayout,
	entries: [{
		binding: 0,
		resource: { buffer: uniformBuffer }
	}]
});

const vertexBufferLayout: GPUVertexBufferLayout = {
	attributes: [{
		format: "float32x2",
		offset: 0,
		shaderLocation: 0
	}, {
		format: "float32x2",
		offset: Float32Array.BYTES_PER_ELEMENT * 2,
		shaderLocation: 1
	}],
	arrayStride: Float32Array.BYTES_PER_ELEMENT * 4
};

const vertices = new Float32Array([
//   X     Y     U     V
	-1.0, -1.0,  0.0,  0.0,
	-1.0,  1.0,  0.0,  1.0,
	 1.0, -1.0,  1.0,  0.0,
	
	 1.0, -1.0,  1.0,  0.0,
	-1.0,  1.0,  0.0,  1.0,
	 1.0,  1.0,  1.0,  1.0,
]);

const vertexBuffer = device.createBuffer({
	label: "VBO",
	size: vertices.byteLength,
	usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(vertexBuffer, 0, vertices);

const renderPipelineLayout = device.createPipelineLayout({
	label: "Pipeline layout",
	bindGroupLayouts: [bindGroupLayout]
});

const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
	label: "Pipeline",
	layout: renderPipelineLayout,
	vertex: {
		module: shaderModule,
		entryPoint: "vs",
		buffers: [vertexBufferLayout]
	},
	fragment: {
		module: shaderModule,
		entryPoint: "fs",
		targets: [{ format: canvasFormat }]
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
	vec3.fromValues(0, -2, 2),
	vec3.fromValues(0, 0, 0),
	canvas,
);

camera.onChange = () => {
	device.queue.writeBuffer(
		uniformBuffer,
		uniforms.views.view.byteOffset,
		camera.getViewMatrix() as Float32Array
	);

	device.queue.writeBuffer(
		uniformBuffer,
		uniforms.views.position.byteOffset,
		camera.position as Float32Array
	)
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
	renderPass.setBindGroup(0, bindGroup);
	renderPass.setVertexBuffer(0, vertexBuffer);
	
	renderPass.draw(vertices.byteLength / vertexBufferLayout.arrayStride);
	
	renderPass.end();
	device.queue.submit([encoder.finish()]);
	requestAnimationFrame(frame);
}

const onResize = () => {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;

	const projection = mat4.perspective(
		utils.degToRad(60),
		canvas.width / canvas.height,
		0.01, 1000
	);
	
	device.queue.writeBuffer(
		uniformBuffer,
		uniforms.views.projection.byteOffset,
		projection as Float32Array
	);

	depthTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		format: renderPipelineDescriptor.depthStencil!.format!,
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});
}

window.onresize = onResize;
onResize();
requestAnimationFrame(frame);