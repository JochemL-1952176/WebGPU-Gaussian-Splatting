import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { mat4, utils, vec2, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from './camera.js';
import readPLY from './readPLY.js';
import shaderCode from './shaders/gsplat.wgsl?raw'

async function loadGaussianData(ply: Blob) {
	console.debug(`Begin reading Gaussian data`);
	let begin = performance.now();
	
	const {header, data: dataView} = await readPLY(ply);
	
	let end = performance.now();
	const count = header.vertexCount;
	console.debug(`Read ${count} points in ${((end - begin) / 1000).toFixed(2)}s`);
	console.debug(header);
	
	let filteredProperties = {
		stride: 0,
		properties: {} as Record<string, {
			readOffset: number,
			writeIndex: number,
			operation: (((x: number) => number) | null)
		}>
	};
	
	const colorsperSH = 3;
	const restSHCoeffs = Object.keys(header.properties).filter((v) => v.startsWith('f_rest_')).length / colorsperSH;
	let baseRestIdx = 0;
	for (const [name, {offset}] of Object.entries(header.properties)) {
		// Ignore unused normal data
		if (name.startsWith('n')) continue;

		let writeIndex = filteredProperties.stride;

		// For some reason, the sh components are stored as rrr...ggg...bbb
		// We convert this to rgbrgbrgb... so that we can take a vec3 from contiguous memory in the vs

		if (name.startsWith('f_rest_')) {
			if (name === 'f_rest_0') { baseRestIdx = writeIndex; }
			const restIdx = parseInt(name.split('_')[2]);
			const SHIdx = restIdx % restSHCoeffs;
			const colorIdx = Math.floor(restIdx / restSHCoeffs);
			writeIndex = baseRestIdx + SHIdx * colorsperSH + colorIdx;
		}

		filteredProperties.properties[name] = {
			readOffset: offset,
			writeIndex,
			operation: null
		};
		filteredProperties.stride++;
		
		switch (name) {
			case 'opacity':
				filteredProperties.properties[name].operation =
					(x) => 1 / (1 + Math.exp(-x)); // logit -> linear
				break;
			case 'scale_0':
			case 'scale_1':
			case 'scale_2':
				filteredProperties.properties[name].operation =
					Math.exp; // log -> linear
				break;
			default: break;
		};
	}
	
	const gaussianBuffer = device.createBuffer({
		label: "Gaussian data buffer",
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		size: count * filteredProperties.stride * Float32Array.BYTES_PER_ELEMENT,
		mappedAtCreation: true
	});

	console.debug(`Begin uploading data`);
	begin = performance.now();
	
	{
		const writeBuffer = new Float32Array(gaussianBuffer.getMappedRange());
		const isLittleEndian = header.format === 'binary_little_endian';
		const propertyValues = Object.values(filteredProperties.properties)
		for (let i = 0; i < count; i++) {
			const vertexIndex = i * filteredProperties.stride;
			const vertexReadOffset = i * header.stride;
			for (const {readOffset, writeIndex, operation} of propertyValues) {
				let value = dataView.getFloat32(vertexReadOffset + readOffset, isLittleEndian);
				if (operation) value = operation(value);
				writeBuffer[vertexIndex + writeIndex] = value;
			}
		}
	}

	gaussianBuffer.unmap();
	end = performance.now();
	console.debug(`uploading ${(gaussianBuffer.size / (Math.pow(1024, 3))).toFixed(2)}GiB took ${((end - begin) / 1000).toFixed(2)}s`);
	
	return {count, gaussianBuffer}
};

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
	format: canvasFormat
});

const shaderModule = device.createShaderModule({ code: shaderCode });
const shaderData = makeShaderDataDefinitions(shaderCode);
const cameraUniforms = makeStructuredView(shaderData.structs.Camera);

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
	}]
});

let {count: numGaussians, gaussianBuffer} = await fetch('/default.ply')
	.then(async x => await loadGaussianData(await x.blob()))

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
			// blend: {
			// 	alpha: {
			// 		srcFactor: 'one',
			// 		dstFactor: 'one-minus-src-alpha'
			// 	},
			// 	color: {
			// 		srcFactor: 'src-alpha',
			// 		dstFactor: 'one-minus-src-alpha'
			// 	}
			// }
		}]
	},
	depthStencil: {
		depthWriteEnabled: true,
		format: "depth24plus",
		depthCompare: "less",
	}
};

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
	camera.getViewMatrix(cameraUniforms.views.view);
	cameraUniforms.set({ position: camera.position });
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

const frame = () => {
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
	
	renderPass.draw(4, numGaussians);
	
	renderPass.end();
	device.queue.submit([encoder.finish()]);
	requestAnimationFrame(frame);
};

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

	depthTexture.destroy();
	depthTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		format: renderPipelineDescriptor.depthStencil!.format!,
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});
};

window.onresize = () => {
	console.debug("Resizing");
	onResize();
};

onResize();
requestAnimationFrame(frame);

let lastTime = performance.now() / 1000;
setInterval(() => {
	const time = performance.now() / 1000;
	camera.update(time - lastTime);
	lastTime = time
}, 1000/ 60);

const plyInput = document.getElementById("PLYinput") as HTMLInputElement;
plyInput.addEventListener("change", async (_) => {
	if (plyInput.files) {
		const fileType = plyInput.files[0].name.split('.').pop()!;
		if (fileType === "ply") {
			const {count, gaussianBuffer} = await loadGaussianData(plyInput.files[0]);
			numGaussians = count;
			GaussianBindGroupDescriptor.entries[0].resource.buffer = gaussianBuffer;
			gaussianBindGroup = device.createBindGroup(GaussianBindGroupDescriptor);
		} else console.error(`Filetype ${fileType} is not supported`);
	} else console.error("Error reading dropped files");
});