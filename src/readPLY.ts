type Format = 'char' | 'uchar' | 'int8' | 'uint8' | 'short' | 'ushort' | 'int16' | 'uint16' | 'int' | 'uint' | 'int32' | 'uint32' | 'float' | 'float32' | 'double' | 'float64'
type Properties = {
	[key: string]: Format
}

interface PLYHeader {
	vertexCount: number,
	endianness: 'big' | 'little',
	properties: Properties
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

function readBinaryValue(view: DataView, format: Format, offset: number, littleEndian: boolean) {
	switch (format) {
		case 'char':
		case 'int8':
			return view.getInt8(offset);
		case 'uchar':
		case 'uint8':
			return view.getUint8(offset);
		case 'short':
		case 'int16':
			return view.getInt16(offset);
		case 'ushort':
		case 'uint16':
			return view.getUint16(offset);
		case 'int':
		case 'int32':
			return view.getInt32(offset, littleEndian);
		case 'uint':
		case 'uint32':
			return view.getUint32(offset, littleEndian);
		case 'float':
		case 'float32':
			return view.getFloat32(offset, littleEndian);
		case 'double':
		case 'float64':
			return view.getFloat64(offset, littleEndian);
		default:
			throw Error("Unsupported type");
	}
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
		properties: {}
	};

	for (let line of headerString.split('\n')) {
		line = line.trim();

		if (line.startsWith('format')) {
			const format = line.split(' ')[1];
			header.endianness = format === 'binary_little_endian' ? 'little' : 'big';
		} else if (line.startsWith('element vertex')) {
			header.vertexCount = parseInt(line.split(' ')[2]);
		} else if (line.startsWith('property')) {
			let [propertyType, propertyName] = line.split(' ').slice(1) as [Format, string];
			header.properties[propertyName] = propertyType;
		}
	}

	return { header, dataOffset: headerString.length + 1 };
}

export default async function readPLY(file: File) {
	const propertyFilter = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];

	const data = await file.arrayBuffer();
	const {header, dataOffset} = parsePLYHeader(data);
	const dataView = new DataView(data!, dataOffset);

	const readOffsets: ({offset: number, type: Format})[] = [];
	let stride = 0;
	
	for (const property in header.properties) {
		const propertyType = header.properties[property];
		const size = formatSize(propertyType);
		if (propertyFilter.includes(property)) {
			readOffsets.push({ offset: stride, type: propertyType });
		}
		stride += size;
	}
	
	let pointIndex = 0;
	const pointData = new Float32Array(header.vertexCount * readOffsets.length);

	for (let i = 0; i < header.vertexCount; i++) {
		for (let j = 0; j < readOffsets.length; j++) {
			const { offset, type } = readOffsets[j];
			const value = readBinaryValue(dataView, type, i * stride + offset, header.endianness === 'little');
			pointData[pointIndex++] = value;
		}
	}

	return {header, data: pointData};
} 