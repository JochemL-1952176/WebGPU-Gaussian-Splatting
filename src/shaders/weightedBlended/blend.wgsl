struct VertexOut {
	@builtin(position) pos: vec4f
}

const fullscreenTri = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); 
@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
	var output	: VertexOut;
	output.pos = vec4(fullscreenTri[vertexIndex], 0, 1);
	return output;
}

@group(0) @binding(0) var accumulate: texture_2d<f32>;
@group(0) @binding(1) var revealage: texture_2d<f32>;

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
	var accum = textureLoad(accumulate, vec2<u32>(in.pos.xy), 0);
	let r = textureLoad(revealage, vec2<u32>(in.pos.xy), 0).r;

	return vec4f(accum.rgb / clamp(accum.a, 1e-4, 5e4), r);
}