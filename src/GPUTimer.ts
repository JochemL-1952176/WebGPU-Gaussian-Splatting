type PassDescriptor = GPURenderPassDescriptor | GPUComputePassDescriptor;
type PassEncoder = GPURenderPassEncoder | GPUComputePassEncoder;

export default class GPUTimer {
	#device: GPUDevice;
	#canTimestamp: boolean;
	#querySet: GPUQuerySet;
	#resolveBuffer: GPUBuffer;
	#resultBuffer: GPUBuffer | undefined;
	#resultBuffers: Array<GPUBuffer> = [];
	#numEvents: number;

	constructor(device: GPUDevice, numEvents: number) {
		this.#device = device;
		this.#numEvents = numEvents;
		this.#canTimestamp = device.features.has('timestamp-query');
		this.#querySet = device.createQuerySet({
			type: 'timestamp',
			count: numEvents * 2,
		});

		this.#resolveBuffer = device.createBuffer({
			size: this.#querySet.count * 8,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});
	}

	#beginTimestampPass(encoder: GPUCommandEncoder, fnName: string, descriptor: PassDescriptor, eventIndex: number): PassEncoder {
		if (!this.#canTimestamp) return encoder[fnName](descriptor);
		console.assert(eventIndex < this.#numEvents, `event ${eventIndex} does not exist`);

		const pass: PassEncoder = encoder[fnName]({
			...descriptor,
			timestampWrites: {
				querySet: this.#querySet,
				beginningOfPassWriteIndex: 2 * eventIndex,
				endOfPassWriteIndex: 2 * eventIndex + 1,
			}
		});

		const resolve = () => this.#resolveTiming(encoder);
		const originalEnd = pass.end;
		pass.end = () => {
			originalEnd.call(pass);
			resolve();
		};

		return pass;
	}

	beginRenderPass(encoder: GPUCommandEncoder, descriptor: GPURenderPassDescriptor, eventIndex: number) {
		return this.#beginTimestampPass(encoder, 'beginRenderPass', descriptor, eventIndex) as GPURenderPassEncoder;
	}

	beginComputePass(encoder: GPUCommandEncoder, descriptor: GPUComputePassDescriptor, eventIndex: number) {
		return this.#beginTimestampPass(encoder, 'beginComputePass', descriptor, eventIndex) as GPUComputePassEncoder;
	}

	#resolveTiming(encoder: GPUCommandEncoder) {
		if (!this.#canTimestamp) return;

		this.#resultBuffer = this.#resultBuffers.pop() ?? this.#device.createBuffer({
			size: this.#resolveBuffer.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		encoder.resolveQuerySet(this.#querySet, 0, this.#querySet.count, this.#resolveBuffer, 0);
		encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#resultBuffer, 0, this.#resultBuffer.size);
	}

	async getResults() {
		if (!this.#canTimestamp) return;

		const resultBuffer = this.#resultBuffer!;
		const results: Array<number> = []

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const times = new BigInt64Array(resultBuffer.getMappedRange());
		for (let i = 0; i < this.#numEvents; i++) {
			const duration = Number(times[2 * i + 1] - times[2 * i]) / 1000;
			results.push(duration);
		}
		resultBuffer.unmap();
		this.#resultBuffers.push(resultBuffer);
		return results;
	}

	destroy() {
		this.#querySet.destroy();
		this.#resolveBuffer.destroy();
		this.#resultBuffers.forEach((buffer: GPUBuffer) => buffer.destroy());
	}
}