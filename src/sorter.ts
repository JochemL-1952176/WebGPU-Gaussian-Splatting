import sharedShaderCode from './shaders/shared.wgsl?raw';
import sortShaderCode from './shaders/sort.wgsl?raw';
import Scene from "./scene";
import Renderer from './renderer';
import GPUTimer from './GPUTimer';

export default class Sorter {
	entryBufferA: GPUBuffer;
	#entryBufferB: GPUBuffer;
	#globalSortingBuffer: GPUBuffer;
	#sortingPassBuffers: Array<GPUBuffer>;
	#sortBindGroups: Array<GPUBindGroup>;
	#histogramPipeline: GPUComputePipeline;
	#prefixSumPipeline: GPUComputePipeline;
	#binningPipeline: GPUComputePipeline;

	constructor(device: GPUDevice, renderer: Renderer, scene: Scene) {
		const entryBufferDescriptor: GPUBufferDescriptor = {
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			size: scene.splats.count * Uint32Array.BYTES_PER_ELEMENT * 2
		};

		this.entryBufferA = device.createBuffer(entryBufferDescriptor);
		this.#entryBufferB = device.createBuffer(entryBufferDescriptor);

		let sortCode = sharedShaderCode + sortShaderCode;
		sortCode = sortCode.replace("MAX_TILE_COUNT_BINNING", Math.max(1, Math.floor(scene.splats.count / 1024)).toString());
		const sortShader = device.createShaderModule({ code: sortCode });

		this.#globalSortingBuffer = device.createBuffer({
			size: (256 * (Math.max(1, Math.floor(scene.splats.count / 1024)) + 4) + 5) * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
		});

		this.#sortingPassBuffers = Array.from(Array(4).keys(), (_, pass_index) => {
			const buffer = device.createBuffer({
				size: 4 * Uint32Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.UNIFORM,
				mappedAtCreation: true
			});

			const localBuffer = new Uint32Array(buffer.getMappedRange());
			localBuffer[0] = pass_index;
			buffer.unmap();
			return buffer;
		});

		const sortBindGroupLayout = device.createBindGroupLayout({
			label: "Sort bind group layout",
			entries: [
				renderer.cameraUniformsLayoutEntry,
				renderer.gaussiansLayoutEntry,
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" }
				}, {
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" }
				}, {
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" }
				}, {
					binding: 5,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" }
				}
			]
		});

		this.#sortBindGroups = Array.from(Array(4).keys(), (_, pass_index) => {
			return device.createBindGroup({
				layout: sortBindGroupLayout,
				entries: [{
					binding: renderer.cameraUniformsLayoutEntry.binding,
					resource: { buffer: renderer.cameraUniformsBuffer }
					}, {
						binding: renderer.gaussiansLayoutEntry.binding,
						resource: { buffer: scene.splats.buffer }
					}, {
						binding: 2,
						resource: { buffer: this.#sortingPassBuffers![pass_index] },
					}, {
						binding: 3,
						resource: { buffer: this.#globalSortingBuffer! },
					}, {
						binding: 4,
						resource: { buffer: pass_index % 2 === 0 ? this.entryBufferA! : this.#entryBufferB! }
					}, {
						binding: 5,
						resource: { buffer: pass_index % 2 === 0 ? this.#entryBufferB! : this.entryBufferA! }
					}
				],
			})
		});

		const sortPipelineLayout = device.createPipelineLayout({
			label: "Sort pipeline layout",
			bindGroupLayouts: [sortBindGroupLayout]
		});

		this.#histogramPipeline = device.createComputePipeline({
			layout: sortPipelineLayout,
			compute: {
				module: sortShader,
				entryPoint: "histogram"
			}
		});

		this.#prefixSumPipeline = device.createComputePipeline({
			layout: sortPipelineLayout,
			compute: {
				module: sortShader,
				entryPoint: "prefixSum"
			}
		});

		this.#binningPipeline = device.createComputePipeline({
			layout: sortPipelineLayout,
			compute: {
				module: sortShader,
				entryPoint: "bin"
			}
		});
	}

	sort(encoder: GPUCommandEncoder, scene: Scene, timer?: GPUTimer) {
		encoder.clearBuffer(this.#globalSortingBuffer);

		const computePass = 
			timer?.beginComputePass(encoder, {}, 0) as GPUComputePassEncoder ??
			encoder.beginComputePass();
		
		const workgroup_entries_a = 4096;
		computePass.setBindGroup(0, this.#sortBindGroups![1]);
		computePass.setPipeline(this.#histogramPipeline);
		computePass.dispatchWorkgroups((scene.splats.count + workgroup_entries_a - 1) / workgroup_entries_a);
		
		// computePass.setPipeline(this.#prefixSumPipeline!);
		// computePass.dispatchWorkgroups(1, 4);
		
		computePass.end();
		
		// this.#sortBindGroups!.forEach((bindgroup, pass_index) => {
		// 	// clear status counters
		// 	if (pass_index > 0)
		// 		encoder.clearBuffer(this.#globalSortingBuffer!, 0, 4 * Math.max(1, Math.floor(scene.splats.count / 1024)) * Uint32Array.BYTES_PER_ELEMENT);
		
		// 	const computePass = 
		// 		timer?.beginComputePass(encoder, {}, pass_index + 1) as GPUComputePassEncoder ??
		// 		encoder.beginComputePass();
			
		// 	computePass.setPipeline(this.#binningPipeline!);
		// 	computePass.setBindGroup(0, bindgroup);
		// 	computePass.dispatchWorkgroups((scene.splats.count + workgroup_entries_a - 1) / workgroup_entries_a);
		// 	computePass.end();
		// })
	}

	destroy() {
		this.entryBufferA?.destroy();
		this.#entryBufferB?.destroy();
		this.#globalSortingBuffer?.destroy();
		this.#sortingPassBuffers?.forEach((buffer) => buffer.destroy());
	}
}