@group(1) @binding(0) var<storage, read> sorted_entries: array<Entry>;

struct VertexOut {
	@builtin(position) pos: vec4f,
	@location(0) color: vec3f,
	@location(1) uv: vec2f,
	@location(2) conic: vec3f,
	@location(3) opacity: f32,
};

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
	if (alpha < 1.0 / 255) { discard; }

	return vec4f(in.color, alpha);
}