import sharedShaderCode from './shaders/shared.wgsl?raw';
import sortShaderCode from './shaders/sort.wgsl?raw';
import Scene from "./scene";
import Renderer from './renderer';
import GPUTimer from './GPUTimer';
import { GPUSplats } from './loadGaussians';

const workgroup_entries_a = 4096;
const workgroup_entries_c = 1024;

export default class SplatSorter {
	entryBufferA: GPUBuffer;
	drawIndirectBuffer: GPUBuffer;

	#entryBufferB: GPUBuffer;

	#sortingPassBuffers: Array<GPUBuffer>;
	#globalSortingBuffer: GPUBuffer;
	#statusCounterBuffer: GPUBuffer;

	#sortBindGroups: Array<GPUBindGroup>;
	#histogramPipeline: GPUComputePipeline;
	#prefixSumPipeline: GPUComputePipeline;
	#binningPipeline: GPUComputePipeline;

	constructor(device: GPUDevice, renderer: Renderer, splats: GPUSplats) {
		const entryBufferDescriptor: GPUBufferDescriptor = {
			usage: GPUBufferUsage.STORAGE,
			size: splats.count * Uint32Array.BYTES_PER_ELEMENT * 2
		};

		this.entryBufferA = device.createBuffer(entryBufferDescriptor);
		this.#entryBufferB = device.createBuffer(entryBufferDescriptor);

		let sortCode = sharedShaderCode + sortShaderCode;
		const TileCountBinning = Math.round((splats.count + workgroup_entries_c - 1) / workgroup_entries_c);
		sortCode = sortCode.replace("MAX_TILE_COUNT_BINNING", TileCountBinning.toString());
		const sortShader = device.createShaderModule({ code: sortCode });

		this.#sortingPassBuffers = Array.from(Array(4).keys(), (_, pass_index) => {
			const buffer = device.createBuffer({
				size: Uint32Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.UNIFORM,
				mappedAtCreation: true
			});

			const localBuffer = new DataView(buffer.getMappedRange());
			localBuffer.setUint32(0, pass_index, true);
			buffer.unmap();
			return buffer;
		});

		this.#globalSortingBuffer = device.createBuffer({
			size: 1025 * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.#statusCounterBuffer = device.createBuffer({
			size: 256 * TileCountBinning * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.drawIndirectBuffer = device.createBuffer({
			size: 4 * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
		});

		const computeStorageLayoutEntry: GPUBindGroupLayoutEntry = {
			binding: -1,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "storage" }
		};

		const sortBindGroupLayout = device.createBindGroupLayout({
			label: "Sort bind group layout",
			entries: [
				renderer.cameraUniformsLayoutEntry,
				renderer.gaussiansLayoutEntry,
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform"}
				}, {
					...computeStorageLayoutEntry,
					binding: 3
				}, {
					...computeStorageLayoutEntry,
					binding: 4
				}, {
					...computeStorageLayoutEntry,
					binding: 5
				}, {
					...computeStorageLayoutEntry,
					binding: 6
				}, {
					...computeStorageLayoutEntry,
					binding: 7
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
						resource: { buffer: splats.buffer }
					}, {
						binding: 2,
						resource: { buffer: this.#sortingPassBuffers[pass_index] },
					}, {
						binding: 3,
						resource: { buffer: this.#globalSortingBuffer },
					}, {
						binding: 4,
						resource: { buffer: this.#statusCounterBuffer },
					}, {
						binding: 5,
						resource: { buffer: this.drawIndirectBuffer },
					}, {
						binding: 6,
						resource: { buffer: pass_index % 2 === 0 ? this.entryBufferA : this.#entryBufferB }
					}, {
						binding: 7,
						resource: { buffer: pass_index % 2 === 0 ? this.#entryBufferB : this.entryBufferA }
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
				entryPoint: "exPrefixSum"
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

	async sort(encoder: GPUCommandEncoder, scene: Scene, timer: GPUTimer) {
		encoder.clearBuffer(this.#globalSortingBuffer);
		encoder.clearBuffer(this.drawIndirectBuffer);

		let computePass = timer.beginComputePass(encoder, {}, 0);
			
		computePass.setBindGroup(0, this.#sortBindGroups![1]);
		computePass.setPipeline(this.#histogramPipeline);
		computePass.dispatchWorkgroups((scene.splats.count + workgroup_entries_a - 1) / workgroup_entries_a);

		computePass.setPipeline(this.#prefixSumPipeline);
		computePass.dispatchWorkgroups(1, 4);
		computePass.end();
		
		this.#sortBindGroups!.forEach((bindgroup, pass_index) => {
			encoder.clearBuffer(this.#statusCounterBuffer);
			const computePass = timer.beginComputePass(encoder, {}, pass_index + 1);
			
			computePass.setPipeline(this.#binningPipeline);
			computePass.setBindGroup(0, bindgroup);
			computePass.dispatchWorkgroups((scene.splats.count + workgroup_entries_c - 1) / workgroup_entries_c);
			computePass.end();
		});
	}

	destroy() {
		this.entryBufferA?.destroy();
		this.#entryBufferB?.destroy();
		this.#sortingPassBuffers?.forEach((buffer) => buffer.destroy());
		this.#globalSortingBuffer?.destroy();
		this.drawIndirectBuffer?.destroy();
		this.#statusCounterBuffer?.destroy();
	}
}