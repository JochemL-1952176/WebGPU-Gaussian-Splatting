import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { mat4, utils, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from './camera.js';
import readPLY from './readPLY.js';

if (!navigator.gpu) {
	throw new Error("WebGPU not supported in this browser");
}


const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
	throw new Error("No GPUAdapter found");
}

/*
* The maximum buffer size as reported by adapter.limits is only 256MiB,
* which would only allow for a few million gaussians. This limit seems
* arbitrarily chosen and doesn't seem to be hardware-related, so as long as the
* device physically supports the requested amount of memory, the below code 
* can be used to bypass this limit.
*/
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: 536870912 }});
const canvas = document.querySelector("canvas")!;
const context = canvas.getContext("webgpu")!;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat,
	alphaMode: 'premultiplied'
});

console.log(device.limits);

const shaderCode = `
const SH_dc = 0.28209479177387814f;

struct Camera {
	position: vec3f,
	view: mat4x4f,
	projection: mat4x4f
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexIn {
	@location(0) pos: vec3f,
	@location(1) color: vec3f,
	@location(2) scale: vec3f,
	@location(3) rotation: vec4f
};

struct FragmentIn {
	@builtin(position) pos: vec4f,
	@location(0) color: vec3f,
	@location(2) uv: vec2f
};

@vertex fn vs(in: VertexIn, @builtin(vertex_index) vertexIndex: u32) -> FragmentIn {
	let quad = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 1));
	let size = 0.0015;

	let uv = quad[vertexIndex];
	let offset = (uv - 0.5) * size * 2.0;
	let viewPos = (cam.view * vec4f(in.pos, 1)) + vec4f(offset, 0, 0);

	return FragmentIn(
		cam.projection * viewPos,
		0.5 + SH_dc * in.color,
		uv
	);
}

@fragment fn fs(in: FragmentIn) -> @location(0) vec4f {
	let d = length(in.uv - vec2f(0.5));
	if (d > 0.5) { discard; }
	return vec4f(in.color, 1);
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

const pointStride = 13;
const pointDataLayout: GPUVertexBufferLayout = {
	stepMode: 'instance',
	attributes: [{
		shaderLocation: 0,
		offset: 0,
		format: "float32x3"
	}, {
		shaderLocation: 1,
		offset: 3 * Float32Array.BYTES_PER_ELEMENT,
		format: 'float32x3'
	}, {
		shaderLocation: 2,
		offset: 6 * Float32Array.BYTES_PER_ELEMENT,
		format: 'float32x3'
	}, {
		shaderLocation: 3,
		offset: 9 * Float32Array.BYTES_PER_ELEMENT,
		format: 'float32x4'
	}],
	arrayStride: pointStride * Float32Array.BYTES_PER_ELEMENT
};

let numPoints = 0;
let pointsBuffer = device.createBuffer({
	usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
	size: 0
});

const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
	label: "Pipeline",
	layout: device.createPipelineLayout({
		label: "Pipeline layout",
		bindGroupLayouts: [bindGroupLayout]
	}),
	primitive: { topology: 'triangle-strip' },
	vertex: {
		module: shaderModule,
		entryPoint: "vs",
		buffers: [pointDataLayout]
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
	vec3.fromValues(0, 0, 1),
	vec3.fromValues(0, 0, 0),
	canvas,
);

camera.onChange = () => {
	uniforms.set({ view: camera.getViewMatrix(), position: camera.position });

	device.queue.writeBuffer(
		uniformBuffer,
		uniforms.views.position.byteOffset,
		uniforms.arrayBuffer,
		uniforms.views.position.byteOffset,
		uniforms.views.projection.byteOffset
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
	renderPass.setBindGroup(0, bindGroup);
	renderPass.setVertexBuffer(0, pointsBuffer);
	
	renderPass.draw(4, numPoints);
	
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

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
	document.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
});

document.ondrop = async (e) => {
	const files = e.dataTransfer?.files;
	if (files) {
		const begin = performance.now();
		console.log(`Begin reading file: ${files[0].name}`);
		const {header, data} = await readPLY(files[0]);
		const end = performance.now();
		numPoints = header.vertexCount;
		console.log(`Read ${numPoints} points in ${((end - begin) / 1000).toFixed(2)}s`);

		pointsBuffer = device.createBuffer({
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
			size: data.byteLength
		});

		device.queue.writeBuffer(pointsBuffer, 0, data);
	}
}