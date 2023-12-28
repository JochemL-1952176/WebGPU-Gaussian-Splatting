type Format = 'char' | 'uchar' | 'int8' | 'uint8' | 'short' | 'ushort' | 'int16' | 'uint16' | 'int' | 'uint' | 'int32' | 'uint32' | 'float' | 'float32' | 'double' | 'float64'
type Properties = {
	[key: string]: {
		offset: number,
		size: number,
		format: Format
	}
}

function formatSize(format: Format) {
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

interface PLYHeader {
	vertexCount: number,
	endianness: 'big' | 'little',
	properties: Properties,
	stride: number
}

function parsePLYHeader(buffer: ArrayBuffer) {
	let headerString = '';
	
	const ENDHEADER = 'end_header';
	let headerOffset = 0;
	const decoder = new TextDecoder('utf8');
	while (true) {
		const chunk = new Uint8Array(buffer, headerOffset, 64);
		headerString += decoder.decode(chunk);
		headerOffset += 64;
		if (headerString.includes(ENDHEADER)) {
			headerString = headerString.slice(0, headerString.indexOf(ENDHEADER) + ENDHEADER.length);
			break;
		};
	}

	let header: PLYHeader = {
		vertexCount: 0,
		endianness: 'little',
		properties: {},
		stride: 0
	};

	for (let line of headerString.split('\n')) {
		line = line.trim();

		if (line.startsWith('format')) {
			const format = line.split(' ')[1];
			header.endianness = format === 'binary_little_endian' ? 'little' : 'big';
		} else if (line.startsWith('element vertex')) {
			header.vertexCount = parseInt(line.split(' ')[2]);
		} else if (line.startsWith('property')) {
			const [propertyType, propertyName] = line.split(' ').slice(1) as [Format, string];
			const size = formatSize(propertyType);
			header.properties[propertyName] = {
				size,
				offset: header.stride,
				format: propertyType
			};
			header.stride++;
		}
	}

	return { header, dataOffset: headerString.length + 1 };
}

export default async function readPLY(file: File) {
	const data = await file.arrayBuffer();
	const {header, dataOffset} = parsePLYHeader(data);
	return {header, data: new Float32Array(data.slice(dataOffset))};
} 