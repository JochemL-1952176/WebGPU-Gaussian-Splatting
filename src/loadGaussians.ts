import readPLY from "./readPLY";

export async function loadGaussianData(ply: Blob, device: GPUDevice) {
	console.debug(`Begin reading Gaussian data`);
	let begin = performance.now();
	
	const {header, data: dataView} = await readPLY(ply);
	
	let end = performance.now();
	const count = header.vertexCount;
	console.debug(`Read ${count} points in ${((end - begin) / 1000).toFixed(2)}s`);

	console.assert(header.format === "binary_little_endian");
	console.debug(header);
	
	const filteredProperties = {
		stride: 0,
		properties: {} as Record<string, {
			readOffset: number,
			writeIndex: number,
			operation: (((x: number) => number) | null)
		}>
	};
	
	const colorsperSH = 3;
	const restSHCoeffs = Object.keys(header.properties).filter((v) => v.startsWith('f_rest_')).length / colorsperSH;
	console.assert(restSHCoeffs === 15, `Expected 15 AC SH coefficients, read ${restSHCoeffs}`);

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
	console.debug(`uploading ${(gaussianBuffer.size / (Math.pow(1024, 2))).toFixed(2)}MiB took ${((end - begin) / 1000).toFixed(2)}s`);
	
	return {count, gaussianBuffer}
};