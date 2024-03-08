import { utils, vec2 } from 'wgpu-matrix';
import TrackballControls, { PerspectiveCamera } from './cameraControls';
import debugGaussiansURL from './assets/default.ply?url';
import { loadGaussianData } from './loadGaussians';
import loadCameras from './loadCameras';
import Renderer, { RenderTimings } from './renderer';
import Scene from './scene';
import { Pane } from 'tweakpane';

if (!navigator.gpu) { throw new Error("WebGPU not supported in this browser"); }
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) { throw new Error("No GPUAdapter found"); }

const timestampEnabled = adapter.features.has('timestamp-query');
if (!timestampEnabled)
console.warn("Feature 'timestamp-query' is not enabled, GPU timings will not be available");

const device = await adapter.requestDevice({
	requiredLimits: {
		maxComputeInvocationsPerWorkgroup: 1024,
		maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
		maxBufferSize: 2 * Math.pow(1024, 3) // 2 GiB
	},
	requiredFeatures: timestampEnabled ? ['timestamp-query' as GPUFeatureName] : []
});

const canvas = document.querySelector("canvas")!;
const context = canvas.getContext("webgpu")!;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat,
	alphaMode: "premultiplied"
});

const renderer = new Renderer(device, context);
let scene = await fetch(debugGaussiansURL)
	.then(async response => await response.blob())
	.then(async blob => await loadGaussianData(blob, device))
	.then(async splats => new Scene(device, renderer, splats));

const camera = new PerspectiveCamera(utils.degToRad(60), canvas.width / canvas.height, 0.01, 100, renderer.cameraUniforms);
const controls = new TrackballControls(camera, canvas);

type Telemetry = {
	fps: number,
	frameTime: number,
	jsTime: number,
} & RenderTimings;

const telemetry: Telemetry = {
	fps: 0,
	frameTime: 0,
	jsTime: 0,
	sorting: 0,
	rendering: 0,
};

let accTime = 0;
const updateSlice = 1 / 100; // 100 updates per second
let lastTime = performance.now();

function frame(frameStart: number) {
	requestAnimationFrame(frame);

	const deltaTime = frameStart - lastTime;
	lastTime = frameStart;
	
	// UPDATE
	// Decouple camera control from framerate
	accTime += deltaTime / 1000;
	while (accTime > updateSlice) {
		controls.update();
		accTime -= updateSlice;
	}

	// DRAW
	renderer.renderFrame(device, scene, camera);

	// TELEMETRY
	telemetry.frameTime = deltaTime;
	telemetry.fps = 1000 / telemetry.frameTime;
	telemetry.jsTime = performance.now() - frameStart;
	renderer.getTimings().then((timings: RenderTimings) => {
		telemetry.sorting = timings.sorting;
		telemetry.rendering = timings.rendering;
	});
};

requestAnimationFrame(frame);

function onResize() {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
	renderer.setSize(device, canvas.width, canvas.height);

	console.debug(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

	camera.aspect = canvas.width / canvas.height;
	
	const focalY = canvas.height / (2 * Math.tan(camera.fov / 2));

	const tanHalfFov = vec2.fromValues(canvas.width / (2 * focalY), canvas.height / (2 * focalY));
	const focal = vec2.fromValues(focalY, focalY);
	
	camera.set({tanHalfFov, focal});
	camera.recalculateProjectionMatrix();
};

onResize();
window.onresize = onResize;

//##############################//
//								//
//		UI-related code			//
//								//
//##############################//

const pane = new Pane({
	title: "Settings",
	container: document.getElementById("tpContainer")!
});

const plyInput = document.getElementById("plyInput") as HTMLInputElement;
const plyButton = pane.addButton({ title: "Load gaussian data" });
plyButton.on('click', () => plyInput.click());

plyInput.onchange = async (_) => {
	if (plyInput.files && plyInput.files.length) {
		let currentCameras = scene.cameras;
		const splats = await loadGaussianData(plyInput.files[0], device);
		scene.destroy();
		scene = new Scene(device, renderer, splats);
		scene.cameras = currentCameras;
	}
	plyInput.value = "";
};

const cameraInput = document.getElementById("cameraInput") as HTMLInputElement;
const cameraButton = pane.addButton({ title: "Load camera data" });
cameraButton.on('click', () => cameraInput.click());

const cameraUI = {
	idx: 0,
	name: ""
};

function setActiveCamera(idx: number) {
	if (!scene.cameras) {
		console.error("No cameras loaded");
		return;
	}

	if (idx < 0 || idx >= scene.cameras.length) return;

	cameraUI.idx = idx;
	cameraUI.name = scene.cameras[idx].img_name;

	cameraSelector.refresh();
	controls.setPose(scene.cameras[idx].position, scene.cameras[idx].rotation);
}

cameraInput.onchange = async (_) => {
	if (cameraInput.files && cameraInput.files.length) {
		scene.cameras = await loadCameras(cameraInput.files[0]);

		cameraFolder.hidden = false;
		(cameraSelector as any).min = 0;
		(cameraSelector as any).max = scene.cameras.length - 1;

		setActiveCamera(0);
	}
	cameraInput.value = "";
};

const controlsFolder = pane.addFolder({ title: "controls" });

const SHSlider = controlsFolder.addBinding(renderer.controlsUniforms.views.maxSH, '0', {
	label: 'Spherical harmonics degree',
	min: 0, max: 3, step: 1, value: 3
});

function SHUpdate() {
	device.queue.writeBuffer(
		renderer.controlsUniformsBuffer,
		renderer.controlsUniforms.views.maxSH.byteOffset,
		renderer.controlsUniforms.views.maxSH
	);
}

SHSlider.on('change', SHUpdate);
SHUpdate();

const scaleSlider = controlsFolder.addBinding(renderer.controlsUniforms.views.scaleMod, '0', {
	label: 'Gaussian scale modifier',
	min: 0, max: 1, step: 0.01, value: 1
});

function scaleUpdate() {
	device.queue.writeBuffer(
		renderer.controlsUniformsBuffer,
		renderer.controlsUniforms.views.scaleMod.byteOffset,
		renderer.controlsUniforms.views.scaleMod
	);
}

scaleSlider.on('change', scaleUpdate);
scaleUpdate();

const telemetryFolder = pane.addFolder({ title: "Telemetry" });

telemetryFolder.addBinding(telemetry, 'fps', {
	readonly: true,
	label: 'FPS',
	format: (v: number) => `${Math.round(v)} FPS`
});

telemetryFolder.addBinding(telemetry, 'frameTime', {
	readonly: true,
	view: 'graph',
	label: 'Frametime',
	format: (v: number) => `${v.toFixed(2)}ms`,
	min: 0, max: 100/3
});

telemetryFolder.addBinding(telemetry, 'jsTime', {
	readonly: true,
	view: 'graph',
	label: 'Javascript time',
	format: (v: number) => `${v.toFixed(2)}ms`,
	min: 0, max: 100/3
});

telemetryFolder.addBinding(telemetry, 'sorting', {
	readonly: true,
	view: 'graph',
	label: 'Sort time',
	format: (v: number) => `${v.toFixed(2)}μs`,
	min: 0, max: 100000/3
});

telemetryFolder.addBinding(telemetry, 'rendering', {
	readonly: true,
	view: 'graph',
	label: 'Render time',
	format: (v: number) => `${v.toFixed(2)}μs`,
	min: 0, max: 100000/3
});

const cameraFolder = pane.addFolder({ title: "Camera", hidden: true });

const cameraSelector = cameraFolder.addBinding(cameraUI, 'idx', {
	label: "Camera ID",
	min: 0, max: 1, step: 1
});
cameraSelector.on('change', (e) => setActiveCamera(e.value));
(cameraSelector.element.querySelector(".tp-sldv") as HTMLDivElement)
	.onwheel = (e) => setActiveCamera(cameraUI.idx - Math.sign(e.deltaY));

cameraFolder.addBinding(cameraUI, 'name', {
	label: "Image name",
	readonly: true
});