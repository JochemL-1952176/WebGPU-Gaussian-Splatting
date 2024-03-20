import Scene from '../scene';
import { Camera } from '../cameraControls';
import { FolderApi } from '@tweakpane/core';
import { Pane } from 'tweakpane';
import CommonRendererData from './common';

export abstract class Renderer {
	common: CommonRendererData;

	constructor(_device: GPUDevice, common: CommonRendererData) {
		this.common = common;
	}

	abstract finalize(device: GPUDevice, scene: Scene): void;
	abstract renderFrame(device: GPUDevice, scene: Scene, camera: Camera): void;
	abstract setSize(device: GPUDevice, width: number, height: number): void;
	abstract controlPanes(root: FolderApi | Pane, device: GPUDevice): void;
	abstract telemetryPanes(root: FolderApi | Pane, interval: number): void;
	abstract destroy(): void;
}

export type rendererConstructor<T extends Renderer> = new(device: GPUDevice, common: CommonRendererData, ...args: any[]) => T;
export default class RendererFactory {
	#commonData: CommonRendererData;
	get commonData() { return this.#commonData; };

	constructor(device: GPUDevice, canvasContext: GPUCanvasContext) {
		this.#commonData = new CommonRendererData(device, canvasContext);
	}

	createRenderer<T extends Renderer>(device: GPUDevice, scene: Scene, type: rendererConstructor<T>, ...args: any[]): T {
		const renderer =  new type(device, this.#commonData, ...args);
		renderer.finalize(device, scene);
		return renderer;
	}

	destroy() {
		this.#commonData.destroy();
	}
}