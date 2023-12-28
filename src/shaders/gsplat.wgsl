const SH_dc = 0.28209479177387814f;

struct Camera {
	position: vec3f,
	view: mat4x4f,
	projection: mat4x4f
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct GaussianOffsets {
	stride: u32,
	pos: u32,
	normal: u32,
	opacity: u32,
	scale: u32,
	rotation: u32,
	sh: u32,
};

// Use flat array to bypass struct padding
@group(1) @binding(0) var<storage, read> gaussianData: array<f32>;
@group(1) @binding(1) var<uniform> gaussianOffset: GaussianOffsets;

struct Gaussian {
	pos: vec3f,
	normal: vec3f,
	opacity: f32,
	scale: vec3f,
	rotation: vec4f,
	sh: array<vec3f, 16>
};

fn readVec3(offset: u32) -> vec3f {
	let x = gaussianData[offset];
	let y = gaussianData[offset + 1];
	let z = gaussianData[offset + 2];

	return vec3f(x, y, z);
}

fn readVec4(offset: u32) -> vec4f {
	let x = gaussianData[offset];
	let y = gaussianData[offset + 1];
	let z = gaussianData[offset + 2];
	let w = gaussianData[offset + 3];

	return vec4f(x, y, z, w);
}

fn sigmoid(x: f32) -> f32 {
	return 1 / (1 + exp(-x));
}

fn readGaussian(index: u32) -> Gaussian {
	let baseOffset = index * gaussianOffset.stride;

	var sh = array<vec3f, 16>();
	for (var i: u32 = 0; i < 16; i = i + 1) {
		sh[i] = readVec3(baseOffset + gaussianOffset.sh + i * 12);
	}

	return Gaussian(
		readVec3(baseOffset + gaussianOffset.pos),
		readVec3(baseOffset + gaussianOffset.normal),
		sigmoid(gaussianData[baseOffset + gaussianOffset.opacity]),
		exp2(readVec3(baseOffset + gaussianOffset.scale)),
		readVec4(baseOffset + gaussianOffset.rotation),
		sh
	);
}

struct VertexIn {
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32
};

struct VertexOut {
	@builtin(position) pos: vec4f,
	@location(0) color: vec3f,
	@location(1) opacity: f32,
	@location(2) uv: vec2f
};

@vertex fn vs(in: VertexIn) -> VertexOut {
	let quad = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 1));

	let gaussian = readGaussian(in.instanceIndex);
	
	let size = 0.005;
	let uv = quad[in.vertexIndex];
	let offset = (uv - 0.5) * size * 2.0;
	let viewPos = (cam.view * vec4f(gaussian.pos, 1)) + vec4f(offset, 0, 0);

	return VertexOut(
		cam.projection * viewPos,
		0.5 + SH_dc * gaussian.sh[0],
		gaussian.opacity,
		uv
	);
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
	let d = length(in.uv - vec2f(0.5));
	if (d > 0.5) { discard; }
	return vec4f(in.color, in.opacity);
}