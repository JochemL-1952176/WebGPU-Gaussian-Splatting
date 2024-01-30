export type PropertyType = 'char' | 'uchar' | 'int8' | 'uint8' | 'short' | 'ushort' | 'int16' | 'uint16' | 'int' | 'uint' | 'int32' | 'uint32' | 'float' | 'float32' | 'double' | 'float64'
export type Format = 'ascii' | 'binary_little_endian' | 'binary_big_endian';
export type Properties = Record<string, {offset: number, type: PropertyType}>;

export type PLYHeader = {
	vertexCount: number,
	format: Format,
	properties: Properties,
	stride: number
};

export function formatSize(format: PropertyType) {
	switch (format) {
		case 'char':
		case 'int8':
			return Int8Array.BYTES_PER_ELEMENT;
		case 'uchar':
		case 'uint8':
			return Uint8Array.BYTES_PER_ELEMENT;
		case 'short':
		case 'int16':
			return Int16Array.BYTES_PER_ELEMENT;
		case 'ushort':
		case 'uint16':
			return Uint16Array.BYTES_PER_ELEMENT;
		case 'int':
		case 'int32':
			return Int32Array.BYTES_PER_ELEMENT;
		case 'uint':
		case 'uint32':
			return Uint32Array.BYTES_PER_ELEMENT;
		case 'float':
		case 'float32':
			return Float32Array.BYTES_PER_ELEMENT;
		case 'double':
		case 'float64':
			return Float64Array.BYTES_PER_ELEMENT;
		default:
			throw Error("Unsupported type");
	}
}

function parsePLYHeader(buffer: ArrayBuffer) {
	let headerString = '';
	const chunkSize = 64;
	
	const ENDHEADER = 'end_header';
	let headerOffset = 0;
	const decoder = new TextDecoder('utf8');
	while (true) {
		const chunk = new Uint8Array(buffer, headerOffset, chunkSize);
		headerString += decoder.decode(chunk);
		headerOffset += chunkSize;
		if (headerString.includes(ENDHEADER)) {
			headerString = headerString.slice(0, headerString.indexOf(ENDHEADER) + ENDHEADER.length);
			break;
		};
	}
	console.assert(headerOffset < buffer.byteLength);

	let header: PLYHeader = {
		vertexCount: 0,
		format: 'binary_little_endian',
		properties: {},
		stride: 0
	};

	for (let line of headerString.split('\n')) {
		line = line.trim();

		if (line.startsWith('format')) {
			header.format = line.split(' ')[1] as Format;

		} else if (line.startsWith('element vertex')) {
			header.vertexCount = parseInt(line.split(' ')[2]);

		} else if (line.startsWith('property')) {
			const [propertyType, propertyName] = line.split(' ').slice(1) as [PropertyType, string];
			const size = formatSize(propertyType);

			header.properties[propertyName] = {
				offset: header.stride,
				type: propertyType
			};

			header.stride += size;
		}
	}

	return { header, dataOffset: headerString.length + 1 };
}

export default async function readPLY(file: Blob) {
	const data = await file.arrayBuffer();
	const {header, dataOffset} = parsePLYHeader(data);
	return {header, data: new DataView(data, dataOffset)};
}