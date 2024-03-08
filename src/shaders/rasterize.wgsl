struct RenderControls {
	maxSH: u32,
	scaleMod: f32
};

@group(0) @binding(2) var<uniform> controls: RenderControls;
@group(1) @binding(0) var<storage, read> sorted_entries: array<Entry>;

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

fn computeColorFromSH(pos: vec3f, baseIndex: u32) -> vec3f {
	let shIndex = baseIndex + shOffset;
	var color = Y00 * readVec3(shIndex);

	if (controls.maxSH >= 1) {
		let dir = normalize(pos - cam.position);
		let x = dir.x; let y = dir.y; let z = dir.z;
		color += Y1X * (-y * readVec3(shIndex + 3) + z * readVec3(shIndex + 6) - x * readVec3(shIndex + 9));

		if (controls.maxSH >= 2) {
			let xx = x*x; let yy = y*y; let zz = z*z;
			let xy = x*y; let yz = y*z; let xz = x*z;
			color += Y2[0] * xy * readVec3(shIndex + 12) +
					Y2[1] * yz * readVec3(shIndex + 15) +
					Y2[2] * (2 * zz - xx - yy) * readVec3(shIndex + 18) +
					Y2[3] * xz * readVec3(shIndex + 21) +
					Y2[4] * (xx - yy) * readVec3(shIndex + 24);

			if (controls.maxSH >= 3) {
			color += Y3[0] * y * (3 * xx - yy) * readVec3(shIndex + 27) +
					Y3[1] * xy * z * readVec3(shIndex + 30) +
					Y3[2] * y * (4 * zz - xx - yy) * readVec3(shIndex + 33) +
					Y3[3] * z * (2 * zz - 3 * (xx - yy)) * readVec3(shIndex + 36) +
					Y3[4] * x * (4 * zz - xx - yy) * readVec3(shIndex + 39) +
					Y3[5] * z * (xx - yy) * readVec3(shIndex + 42) +
					Y3[6] * x * (xx - 3 * yy) * readVec3(shIndex + 45);
			}
		}
	}

	return saturate(color + 0.5);
}

fn computeCov3D(scale: vec3f, rotation: vec4f) -> array<f32, 6> {
	var S = mat3x3f();
	S[0][0] = controls.scaleMod * scale.x;
	S[1][1] = controls.scaleMod * scale.y;
	S[2][2] = controls.scaleMod * scale.z;

	let normRot = normalize(rotation);
	let r = normRot.x;
	let x = normRot.y;
	let y = normRot.z;
	let z = normRot.w;

	let R = mat3x3f(
		1.0 - 2 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
		2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
		2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y),
	);

	let M = S * R;
	let Sigma = transpose(M) * M;
	return array(
		Sigma[0][0],
		Sigma[0][1],
		Sigma[0][2],
		Sigma[1][1],
		Sigma[1][2],
		Sigma[2][2]
	);
}

fn computeCov2D(mean: vec3f, cov3D: array<f32, 6>) -> vec3f {
	let limx = 1.3 * cam.tanHalfFov.x;
	let limy = 1.3 * cam.tanHalfFov.y;

	var t = cam.view * vec4f(mean, 1);
	let txytz = t.xy / t.z;
	t.x = min(limx, max(-limx, txytz.x)) * t.z;
	t.y = min(limy, max(-limy, txytz.y)) * t.z;

	let J = mat3x3f(
		cam.focal.x / t.z, 0, -(cam.focal.x * t.x) / (t.z * t.z),
		0, cam.focal.y / t.z, -(cam.focal.y * t.y) / (t.z * t.z),
		0, 0, 0
	);

	let W = transpose(mat3x3f(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz));
	let T = W * J;

	let Vrk = mat3x3f(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	var cov = transpose(T) * transpose(Vrk) * T;
	cov[0][0] += 0.3;
	cov[1][1] += 0.3;
	return vec3f(cov[0][0], cov[0][1], cov[1][1]);
}

struct VertexIn {
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32
};

struct VertexOut {
	@builtin(position) pos: vec4f,
	@location(0) color: vec3f,
	@location(1) uv: vec2f,
	@location(3) conic: vec3f,
	@location(4) opacity: f32,
};

const quad = array(vec2f(-1, -1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));
@vertex fn vs(in: VertexIn) -> VertexOut {
	var out = VertexOut();
	let entry = sorted_entries[in.instanceIndex];

	let baseIndex = entry.value * stride;
	
	let pos = readVec3(baseIndex + posOffset);

	out.opacity = gaussianData[baseIndex + opacityOffset];
	let scale = readVec3(baseIndex + scaleOffset);
	let rotation = readVec4(baseIndex + rotationOffset);

	let cov2d = computeCov2D(pos, computeCov3D(scale, rotation));
	let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

	if (det == 0) {
		out.pos = vec4f(0);
		return out;
	}

	let det_inv = 1 / det;
	out.conic = vec3f(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

	let mid = 0.5 * (cov2d.x + cov2d.z);
	let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
	let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
	let radius_pix = ceil(3 * sqrt(max(lambda1, lambda2)));
	let wh = 2 * cam.tanHalfFov * cam.focal;
	let radius_ndc = vec2f(radius_pix) / wh;

	let quadOffset = quad[in.vertexIndex];
	var projPos = cam.projection * cam.view * vec4f(pos, 1);
	projPos /= projPos.w;
	
	out.pos = vec4f(projPos.xy + 2 * radius_ndc * quadOffset, projPos.zw);
	out.uv = radius_pix * quadOffset;
	out.color = computeColorFromSH(pos, baseIndex);

	return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
	let d = -in.uv;
	let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
	if (power > 0) { discard; }

	let alpha = min(1, in.opacity * exp(power));

	return vec4f(alpha * in.color, alpha);
}