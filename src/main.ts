import { utils, vec2 } from 'wgpu-matrix';
import TrackballControls, { PerspectiveCamera } from './cameraControls';
import debugGaussiansURL from '@assets/default.ply?url';
import { loadGaussianData } from './loadGaussians';
import loadCameras from './loadCameras';
import RendererFactory, { Renderer, rendererConstructor } from './renderers/renderer';
import Scene from './scene';
import { ListBladeApi, Pane } from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';
import { ClippedRenderer, SortingRenderer } from './renderers';
import { StochasticRenderer } from './renderers/stochastic';

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

const renderers: Record<string, rendererConstructor<Renderer>> = {
	sorted: SortingRenderer,
	clipped:  ClippedRenderer,
	stochastic: StochasticRenderer,
};

let currentRenderer = "sorted";

const rendererFactory = new RendererFactory(device, context); 
let scene = await fetch(debugGaussiansURL)
	.then(async response => await response.blob())
	.then(async blob => await loadGaussianData(blob, device))
	.then(async splats => new Scene(device, rendererFactory.commonData, splats));

let renderer: Renderer = rendererFactory.createRenderer(device, scene, renderers[currentRenderer]);

const camera = new PerspectiveCamera(60, canvas.width / canvas.height, .1, 1000, renderer.common.cameraUniforms);
const controls = new TrackballControls(camera, canvas);

const telemetry = {
	frameTime: 0,
	jsTime: 0
};

let accTime = 0;
const updateSlice = 1 / 100; // 100 updates per second
let lastTime = performance.now();

function frame() {
	const frameStart = performance.now();
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
	telemetry.jsTime = performance.now() - frameStart;

	requestAnimationFrame(frame);
};

function onResize() {
	canvas.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	canvas.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
	renderer.setSize(device, canvas.width, canvas.height);
	imageSize.refresh();

	camera.aspect = canvas.width / canvas.height;
	
	const focalY = canvas.height / (2 * Math.tan(utils.degToRad(camera.fov) / 2));
	const focal = vec2.fromValues(focalY, focalY);
	const tanHalfFov = vec2.fromValues(canvas.width / (2 * focal[0]), canvas.height / (2 * focal[1]));
	
	camera.set({tanHalfFov, focal});
	camera.recalculateProjectionMatrix();
};

window.onresize = onResize;

//##############################//
//								//
//		UI-related code			//
//								//
//##############################//

const GRAPHREFRESHINTERVAL = 50;

const pane = new Pane({
	title: "Settings",
	container: document.getElementById("tpContainer")!
});
pane.registerPlugin(EssentialsPlugin);

const plyInput = document.getElementById("plyInput") as HTMLInputElement;
const cameraInput = document.getElementById("cameraInput") as HTMLInputElement;

(pane.addBlade({
	view: 'buttongrid',
	size: [2, 1],
	cells: (x: number, _y: number) => ({ title: ["Load gaussian data", "Load camera data"][x] })
}) as EssentialsPlugin.ButtonGridApi).on('click', (e) => {
	const [x, _y] = e.index;
	switch (x) {
		case 0: plyInput.click(); break;
		case 1: cameraInput.click(); break;
		default: break;
	}
});

plyInput.onchange = async (_) => {
	if (plyInput.files && plyInput.files.length) {
		const splats = await loadGaussianData(plyInput.files[0], device);
		scene.destroy();
		scene = new Scene(device, rendererFactory.commonData, splats);
		renderer.finalize(device, scene);

		cameraSelection.hidden = true;
	}
	plyInput.value = "";
};

function setActiveCamera(idx: number) {
	console.assert(scene.cameras !== undefined, "No cameras loaded");
	console.assert(idx >= 0 && idx < scene.cameras!.length, "Camera index out of bounds");

	controls.setPose(scene.cameras![idx].position, scene.cameras![idx].rotation);

	const fy = canvas.height * (scene.cameras![idx].fy / scene.cameras![idx].height);
	const fx = canvas.width * (scene.cameras![idx].fx / scene.cameras![idx].width);

	camera.fov = utils.radToDeg(2 * Math.atan(canvas.height / (2 * fy)));

	const focal = vec2.fromValues(fx, fy);
	const tanHalfFov = vec2.fromValues(canvas.width / (2 * focal[0]), canvas.height / (2 * focal[1]));
	
	camera.set({tanHalfFov, focal});
	camera.recalculateProjectionMatrix();
	cameraFolder.refresh();
}

cameraInput.onchange = async (_) => {
	if (cameraInput.files && cameraInput.files.length) {
		scene.cameras = await loadCameras(cameraInput.files[0]);

		cameraSelection.options = scene.cameras.map((cam, idx) => ({ text: cam.img_name, value: idx }));
		cameraSelection.hidden = false;

		setActiveCamera(0);
	}
	cameraInput.value = "";
};

(pane.addBlade({
	view: 'radiogrid',
	groupName: 'renderer',
	size: [Object.keys(renderers).length, 1],
	cells: (x: number, _y: number) => {
		const key = Object.keys(renderers)[x];

		return {
			title: key.charAt(0).toUpperCase() + key.slice(1),
			value: key
		}
	},
	value: currentRenderer
}) as EssentialsPlugin.RadioGridApi<string>).on('change', (e) => {
	currentRenderer = e.value;
	renderer.destroy();
	renderer = rendererFactory.createRenderer(device, scene, renderers[currentRenderer]);
	renderer.finalize(device, scene);
	
	renderer.controlPanes(controlsFolder, device);
	renderer.telemetryPanes(telemetryFolder, GRAPHREFRESHINTERVAL);
});

const controlsFolder = pane.addFolder({ title: "controls" });

const SHSlider = controlsFolder.addBinding(renderer.common.controlsUniforms.views.maxSH, '0', {
	label: 'Spherical harmonics degree',
	min: 0, max: 3, step: 1, value: 3
});

function SHUpdate() {
	device.queue.writeBuffer(
		renderer.common.controlsUniformsBuffer,
		renderer.common.controlsUniforms.views.maxSH.byteOffset,
		renderer.common.controlsUniforms.views.maxSH
	);
}

SHSlider.on('change', SHUpdate);
SHUpdate();

const scaleSlider = controlsFolder.addBinding(renderer.common.controlsUniforms.views.scaleMod, '0', {
	label: 'Gaussian scale modifier',
	min: 0, max: 1, step: 0.01, value: 1
});

function scaleUpdate() {
	device.queue.writeBuffer(
		renderer.common.controlsUniformsBuffer,
		renderer.common.controlsUniforms.views.scaleMod.byteOffset,
		renderer.common.controlsUniforms.views.scaleMod
	);
}

scaleSlider.on('change', scaleUpdate);
scaleUpdate();

const telemetryFolder = pane.addFolder({ title: "Telemetry" });

telemetryFolder.addBinding(telemetry, 'frameTime', {
	readonly: true,
	label: 'Frames per second',
	format: (v: number) => `${Math.round(1000 / v).toString().padStart(3, ' ')} FPS`,
	interval: GRAPHREFRESHINTERVAL
});

telemetryFolder.addBinding(telemetry, 'frameTime', {
	readonly: true,
	view: 'graph',
	label: 'Frametime',
	format: (v: number) => `${v.toFixed(2)}ms`,
	min: 0, max: 100 / 3,
	interval: GRAPHREFRESHINTERVAL
});

telemetryFolder.addBinding(telemetry, 'jsTime', {
	readonly: true,
	view: 'graph',
	label: 'Javascript time',
	format: (v: number) => `${v.toFixed(2)}ms`,
	min: 0, max: 100 / 3,
	interval: GRAPHREFRESHINTERVAL
});

const cameraFolder = pane.addFolder({ title: "Camera" });

const cameraSelection = cameraFolder.addBlade({
	view: 'list',
	label: 'camera',
	hidden: true,
	options: [],
	value: 0
}) as ListBladeApi<number>;

cameraSelection.on('change', (e) => setActiveCamera(e.value));
(cameraSelection.element.querySelector(".tp-lstv") as HTMLDivElement).onwheel = (e) => {
	const upper = scene.cameras?.length ?? 0;
	cameraSelection.value = (((cameraSelection.value - Math.sign(e.deltaY)) % upper) + upper) % upper;
	setActiveCamera(cameraSelection.value);
};

const imageSize = cameraFolder.addBinding(canvas, 'width', {
	label: "Image size",
	format: () => `${canvas.width}x${canvas.height}`,
	readonly: true
});

cameraFolder.addBinding(camera, 'fov', {
	label: "Vertical FOV",
	min: 0, max: 170,
	format: (v: number) => `${v.toFixed(2)}°`
}).on('change', () => {
	const focalY = canvas.height / (2 * Math.tan(utils.degToRad(camera.fov) / 2));
	const focal = vec2.fromValues(focalY, focalY);
	const tanHalfFov = vec2.fromValues(canvas.width / (2 * focal[0]), canvas.height / (2 * focal[1]));
	
	camera.set({tanHalfFov, focal});
	camera.recalculateProjectionMatrix();
});

cameraFolder.addBinding(camera, 'near', {
	label: "Near",
	min: 0
}).on('change', () => camera.recalculateProjectionMatrix());

cameraFolder.addBinding(camera, 'far', {
	label: "Far",
	min: 0
}).on('change', () => camera.recalculateProjectionMatrix());

renderer.controlPanes(controlsFolder, device);
renderer.telemetryPanes(telemetryFolder, GRAPHREFRESHINTERVAL);

onResize();
requestAnimationFrame(frame);