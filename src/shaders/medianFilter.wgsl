struct VertexOut {
	@builtin(position) pos : vec4f
}

const fullscreenTri = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); 
@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
	var output	: VertexOut;
	output.pos = vec4(fullscreenTri[vertexIndex], 0, 1);
	return output;
}

@group(0) @binding(0) var renderResult: texture_2d<f32>;
const halfFilterSize = FILTERSIZE / 2;
const windowSize = FILTERSIZE * FILTERSIZE;

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
	let size = vec2<i32>(textureDimensions(renderResult));
	let center = vec2<i32>(in.pos.xy);
	var window = array<vec3f, windowSize>();
	var windowGray = array<f32, windowSize>();
	var collectedSamples = 0;

	for (var v = -halfFilterSize; v < halfFilterSize; v++) {
		let sampleV = center.y - v;
		if (sampleV < 0 || sampleV >= size.y) { continue; };

		for (var u = -halfFilterSize; u < halfFilterSize; u++) {
			let sampleU = center.x - u;
			if (sampleU < 0 || sampleU >= size.x) { continue; };

			let color = textureLoad(renderResult, center - vec2<i32>(u, v), 0).rgb;
			collectedSamples++;
			window[collectedSamples] = color;
			windowGray[collectedSamples] = rgbToLuma(color);
		}
	}

	for (var i = 0; i < collectedSamples; i++) {
		for (var j = i + 1; j < collectedSamples; j++) {
			if (windowGray[i] > windowGray[j]) {
				{
					let tmp = windowGray[i];
					windowGray[i] = windowGray[j];
					windowGray[j] = tmp;
				}

				{
					let tmp = window[i];
					window[i] = window[j];
					window[j] = tmp;
				}
			}
		}
	}

	if (collectedSamples % 2 == 0) {
		return vec4f((window[collectedSamples / 2] + window[(collectedSamples + 1) / 2]) / 2, 1);
	} else {
		return vec4f(window[collectedSamples / 2], 1);
	}
}

// https://en.wikipedia.org/wiki/Luma_(video)#Rec._601_luma_versus_Rec._709_luma_coefficients
fn rgbToLuma(rgb: vec3f) -> f32 {
	return 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
}