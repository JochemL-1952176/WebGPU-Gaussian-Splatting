struct Camera {
	position: vec3f,
	view: mat4x4f,
	projection: mat4x4f,
	focal: vec2f,
	tan_fov: vec2f
};

struct Entry { key: u32, value: u32 };

@group(0) @binding(0) var<uniform> cam: Camera;

// Use flat array to bypass struct padding
@group(0) @binding(1) var<storage, read> gaussianData: array<f32>;

const stride = 59u;
const posOffset = 0u;
const shOffset = 3u;
const opacityOffset = 51u;
const scaleOffset = 52u;
const rotationOffset = 55u;

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

fn isInFrustum(clipPos: vec3f) -> bool {
    return 	abs(clipPos.x) < 1.3 &&
			abs(clipPos.y) < 1.3 &&
			abs(clipPos.z - 0.5) < 0.5;
}