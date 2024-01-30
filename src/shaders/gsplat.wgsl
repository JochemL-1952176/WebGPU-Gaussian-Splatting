// Adapted from
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer
const PI = radians(180);
const Y00 = 0.5 * sqrt(1 / PI);
const Y1X = sqrt(3 / (4 * PI));
const Y2 = array(
	 0.5 * sqrt(15 / PI),
	-0.5 * sqrt(15 / PI),
	 0.25 * sqrt(5 / PI),
	-0.5 * sqrt(15 / PI),
	 0.25 * sqrt(15 / PI)
);

const Y3 = array(
	-0.25 * sqrt(35 / (2 * PI)),
	 0.5 * sqrt(105 / PI),
	-0.25 * sqrt(21 / (2 * PI)),
	 0.25 * sqrt(7 / PI),
	-0.25 * sqrt(21 / (2 * PI)),
	 0.25 * sqrt(105 / PI),
	-0.25 * sqrt(35 / (2 * PI))
);

const stride = 59;
const posOffset = 0;
const shOffset = 3;
const opacityOffset = 51;
const scaleOffset = 52;
const rotationOffset = 55;

struct Camera {
	position: vec3f,
	view: mat4x4f,
	projection: mat4x4f
};

@group(0) @binding(0) var<uniform> cam: Camera;

// Use flat array to bypass struct padding
@group(1) @binding(0) var<storage, read> gaussianData: array<f32>;

fn readVec3(offset: u32) -> vec3f {
	return vec3f(
		gaussianData[offset],
		gaussianData[offset + 1],
		gaussianData[offset + 2]);
}

fn readVec4(offset: u32) -> vec4f {
	return vec4f(
		gaussianData[offset],
		gaussianData[offset + 1],
		gaussianData[offset + 2],
		gaussianData[offset + 3]);
}

// TODO: only dc works correctly, make it make sense
fn computeColorFromSH(pos: vec3f, baseIndex: u32) -> vec3f {
	let shIndex = baseIndex + shOffset;
	let dir = normalize(pos - cam.position);
	var color = Y00 * readVec3(shIndex);

	// first degree
	let x = dir.x; let y = dir.y; let z = dir.z;
	color += Y1X * (-y * readVec3(shIndex + 3) + z * readVec3(shIndex + 6) - x * readVec3(shIndex + 9));

	// second degree
	let xx = x*x; let yy = y*y; let zz = z*z;
	let xy = x*y; let yz = y*z; let xz = x*z;
	color += Y2[0] * xy * readVec3(shIndex + 12) +
			 Y2[1] * yz * readVec3(shIndex + 15) +
			 Y2[2] * (2 * zz - xx - yy) * readVec3(shIndex + 18) +
			 Y2[3] * xz * readVec3(shIndex + 21) +
			 Y2[4] * (xx - yy) * readVec3(shIndex + 24);

	// third degree
	color += Y3[0] * y * (3 * xx - yy) * readVec3(shIndex + 27) +
			 Y3[1] * xy * z * readVec3(shIndex + 30) +
			 Y3[2] * y * (4 * zz - xx - yy) * readVec3(shIndex + 33) +
			 Y3[3] * z * (2 * zz - 3 * (xx - yy)) * readVec3(shIndex + 36) +
			 Y3[4] * x * (4 * zz - xx - yy) * readVec3(shIndex + 39) +
			 Y3[5] * z * (xx - yy) * readVec3(shIndex + 42) +
			 Y3[6] * x * (xx - 3 * yy) * readVec3(shIndex + 45);

	return saturate(color + 0.5);
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
	const quad = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 1));
	const size = 0.005;

	let baseIndex = in.instanceIndex * stride;
	let pos = readVec3(baseIndex + posOffset);
	let opacity = gaussianData[baseIndex + opacityOffset];
	let scale = readVec3(baseIndex + scaleOffset);
	let rotation = readVec4(baseIndex + rotationOffset);
	
	let uv = quad[in.vertexIndex];
	let offset = (uv - 0.5) * size * 2.0;
	let viewPos = (cam.view * vec4f(pos, 1)) + vec4f(offset, 0, 0);

	return VertexOut(
		cam.projection * viewPos,
		computeColorFromSH(pos, baseIndex),
		opacity,
		uv
	);
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
	let d = length(in.uv - vec2f(0.5));
	if (d > 0.5) { discard; }
	return vec4f(in.color, in.opacity);
}