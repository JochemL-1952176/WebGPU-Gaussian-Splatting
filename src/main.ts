import { utils, vec2 } from 'wgpu-matrix';
import TrackballController, { PerspectiveCamera } from './cameraControls';
import debugGaussiansURL from './assets/default.ply?url';
import { loadGaussianData } from './loadGaussians';
import loadCameras, { Camera } from './loadCameras';
import Renderer from './renderer';
import Scene from './scene';

if (!navigator.gpu) { throw new Error("WebGPU not supported in this browser"); }
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) { throw new Error("No GPUAdapter found"); }

const timestampEnabled = adapter.features.has('timestamp-query');
if (!timestampEnabled)
	console.warn("Feature 'timestamp-query' is not enabled, GPU timings will not be available");

const device = await adapter.requestDevice({
	requiredLimits: {
		maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
		maxBufferSize: 2 * Math.pow(1024, 3) // 2 GiB
	},
	requiredFeatures: [
		...(timestampEnabled ? ['timestamp-query' as GPUFeatureName] : [])
	]
});

const canvas = document.querySelector("canvas")!;
const context = canvas.getContext("webgpu")!;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat,
	alphaMode: "premultiplied"
});

const renderer = new Renderer(device, context, canvasFormat);

let splats = await fetch(debugGaussiansURL)
	.then(async x => await loadGaussianData(await x.blob(), device));

const scene = new Scene(device, renderer, splats);
const camera = new PerspectiveCamera(utils.degToRad(60), canvas.width / canvas.height, 0.01, 1000);
const controller = new TrackballController(canvas);

function updateControllerUniforms() {
	controller.getViewMatrix(renderer.cameraUniforms.views.view);
	renderer.cameraUniforms.set({ position: controller.position });
	const offset = renderer.cameraUniforms.views.position.byteOffset;
	const size = renderer.cameraUniforms.views.projection.byteOffset - offset;

	device.queue.writeBuffer(
		renderer.cameraUniformsBuffer,
		offset,
		renderer.cameraUniforms.arrayBuffer,
		offset,
		size 
	);
}

class Accumulator {
	#sum: number = 0;
	#nSamples: number = 0;

	constructor() {};
	getAverage() { return this.#sum / this.#nSamples;}
	add(v: number) { this.#sum += v; this.#nSamples++; }
	reset() { this.#sum = 0; this.#nSamples = 0; }
}

const frametimeAcc = new Accumulator();
const gputimeAcc = new Accumulator();
const jstimeAcc = new Accumulator();

let accTime = 0;
const updateSlice = 1 / 100; // 100 updates per second

let lastTimingUpdate = performance.now();
const TimingUpdateInterval = 100; //ms
let lastTime = performance.now();

function frame() {
	const frameStart = performance.now();
	const deltaTime = (frameStart - lastTime) / 1000;
	lastTime = frameStart;
	
	// UPDATE
	// Decouple camera control from framerate
	accTime += deltaTime;
	while (accTime > updateSlice) {
		controller.update();
		accTime -= updateSlice;
	}

	if (controller.hasChanged) {
		updateControllerUniforms();
		controller.hasChanged = false;
	}

	// DRAW
	renderer.renderFrame(device, scene);

	// TELEMETRY
	const deltaFPSUpdate = (frameStart - lastTimingUpdate);
	const updateTelemetry = deltaFPSUpdate > TimingUpdateInterval;
	if (updateTelemetry) {
		const frametime = frametimeAcc.getAverage();
		const fps = 1 / frametime;
		const jstime = jstimeAcc.getAverage();
		const gputime = gputimeAcc.getAverage();
		
		frametimeLabel.textContent = `frametime: ${(frametime * 1000).toFixed(1).padStart(4)}ms`;
		fpsLabel.textContent = `${Math.round(fps).toString().padStart(3)} FPS`;
		jstimelabel.textContent = `javascript time: ${jstime.toFixed(1)}ms`
		gputimeLabel.textContent = `render pass time: ${gputime.toFixed(1)}μs`;

		lastTimingUpdate = frameStart;

		frametimeAcc.reset();
		jstimeAcc.reset();
		gputimeAcc.reset();
	}

	renderer.gpuTimer.getResults([(v: number) => gputimeAcc.add(v)]);
	frametimeAcc.add(deltaTime);
	jstimeAcc.add(performance.now() - frameStart);

	requestAnimationFrame(frame);
};

requestAnimationFrame(frame);

function onResize() {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
	renderer.setSize(device, canvas.width, canvas.height);

	console.debug(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

	camera.aspect = canvas.width / canvas.height;
	
	const htany = Math.tan(camera.fov / 2);
	const tan_fov = vec2.fromValues((htany / canvas.height) * canvas.width, htany);
	const focal = vec2.fromValues(canvas.width / (2 * tan_fov[0]), canvas.height / (2 * tan_fov[1]));
	
	renderer.cameraUniforms.set({tan_fov, focal});
	camera.getprojectionMatrix(renderer.cameraUniforms.views.projection);

	const offset = renderer.cameraUniforms.views.projection.byteOffset;
	const size = renderer.cameraUniforms.arrayBuffer.byteLength - offset;

	device.queue.writeBuffer(
		renderer.cameraUniformsBuffer,
		offset,
		renderer.cameraUniforms.arrayBuffer,
		offset,
		size 
	);
};

onResize();
window.onresize = onResize;

//##############################//
//								//
//		UI-related code			//
//								//
//##############################//

const frametimeLabel = document.getElementById("frameTime") as HTMLParagraphElement;
const fpsLabel = document.getElementById("FPS") as HTMLParagraphElement;
const jstimelabel = document.getElementById("jstime") as HTMLParagraphElement;
const gputimeLabel = document.getElementById("gputime") as HTMLParagraphElement;

const SHSlider = document.getElementById("SHslider") as HTMLInputElement;
const SHDisplay = document.getElementById("SHDisplay") as HTMLOutputElement;
function SHUpdate() {
	SHDisplay.value = SHSlider.value;
	const val = parseInt(SHSlider.value);
	renderer.controlsUniforms.set({maxSH: val});
	device.queue.writeBuffer(
		renderer.controlsUniformsBuffer,
		renderer.controlsUniforms.views.maxSH.byteOffset,
		renderer.controlsUniforms.views.maxSH
	);
}

SHUpdate();
SHSlider.oninput = SHUpdate;

const scaleSlider = document.getElementById("scaleSlider") as HTMLInputElement;
const scaleDisplay = document.getElementById("scaleDisplay") as HTMLOutputElement;
function scaleUpdate() {
	const val = parseFloat(scaleSlider.value)
	scaleDisplay.value = val.toFixed(2);
	renderer.controlsUniforms.set({scaleMod: val});
	device.queue.writeBuffer(
		renderer.controlsUniformsBuffer,
		renderer.controlsUniforms.views.scaleMod.byteOffset,
		renderer.controlsUniforms.views.scaleMod
	);
}

scaleUpdate();
scaleSlider.oninput = scaleUpdate;

const plyInput = document.getElementById("PLYinput") as HTMLInputElement;
plyInput.onchange = async (_) => {
	if (plyInput.files && plyInput.files.length) {
		const fileType = plyInput.files[0].name.split('.').pop()!;
		if (fileType === "ply") {
			const splats = await loadGaussianData(plyInput.files[0], device);
			scene.setSplats(device, splats);
		} else console.error(`Filetype ${fileType} is not supported`);
	}
};

let cameras: Array<Camera> | undefined;
let cameraIdx = 0;

const camerasDiv = document.getElementById("cameras") as HTMLDivElement;
const cameraSelector = document.getElementById("cameraID") as HTMLInputElement;
const imgNamelabel = document.getElementById("imageName") as HTMLParagraphElement;

function setActiveCamera(idx: number) {
	if (!cameras) {
		console.error("No cameras loaded");
		return;
	}

	if (idx < 0 || idx >= cameras.length) return;

	cameraIdx = idx;
	controller.setPose(cameras[cameraIdx].position, cameras[cameraIdx].rotation);
	imgNamelabel.textContent = cameras[cameraIdx].img_name;
}

cameraSelector.oninput = () => { setActiveCamera(parseInt(cameraSelector.value)); };

const camerasInput = document.getElementById("Camerainput") as HTMLInputElement;
camerasInput.onchange = async (_) => {
	if (camerasInput.files && camerasInput.files.length) {
		const fileType = camerasInput.files[0].name.split('.').pop()!;
		if (fileType === "json") {
			cameras = await loadCameras(camerasInput.files[0]);
			setActiveCamera(0);

			camerasDiv.style.opacity = "1";
			cameraSelector.min = "0";
			cameraSelector.max = (cameras.length -1).toString();
			cameraSelector.value = cameraSelector.min;
		} else console.error(`Filetype ${fileType} is not supported`);
	}
};