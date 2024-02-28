type PassDescriptor = GPURenderPassDescriptor | GPUComputePassDescriptor;
type PassEncoder = GPURenderPassEncoder | GPUComputePassEncoder;

export default class GPUTimer {
	#canTimestamp: boolean;
	#querySet: GPUQuerySet;
	#resolveBuffer: GPUBuffer;
	#resultBuffer: GPUBuffer;
	#numEvents: number;

	constructor(device: GPUDevice, numEvents: number) {
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

		this.#resultBuffer = device.createBuffer({
			size: this.#resolveBuffer.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
		return this.#beginTimestampPass(encoder, 'beginRenderPass', descriptor, eventIndex);
	}

	beginComputePass(encoder: GPUCommandEncoder, descriptor: GPUComputePassDescriptor, eventIndex: number) {
		return this.#beginTimestampPass(encoder, 'beginComputePass', descriptor, eventIndex);
	}

	#resolveTiming(encoder: GPUCommandEncoder) {
		if (!this.#canTimestamp) return;

		encoder.resolveQuerySet(this.#querySet, 0, this.#querySet.count, this.#resolveBuffer, 0);
		
		if (this.#resultBuffer.mapState === 'unmapped')
			encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#resultBuffer, 0, this.#resultBuffer.size);
	}

	getResults(callbacks: Array<(n: number) => void>) {
		if (!this.#canTimestamp) return;

		if (this.#resultBuffer.mapState === 'unmapped') {
			this.#resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
				const times = new BigInt64Array(this.#resultBuffer.getMappedRange());
				for (let i = 0; i < this.#numEvents; i++) {
					const duration = Number(times[2 * i + 1] - times[2 * i]) / 1000;
					callbacks[i](duration);
				}
				this.#resultBuffer.unmap();
			});
		}
	}
}