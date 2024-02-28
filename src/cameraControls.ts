import { StructuredView } from "webgpu-utils";
import { Mat3, mat4, quat, vec2, vec3, Vec3 } from "wgpu-matrix";

const UP = vec3.fromValues(0, -1, 0);
const FORWARD = vec3.fromValues(0, 0, 1);


export interface Camera {
	uniforms: StructuredView;
	hasChanged: boolean;
	set(data: any): void;
	recalculateProjectionMatrix(): void;
}

export class PerspectiveCamera implements Camera {
	fov: number;
	aspect: number;
	near: number;
	far: number;

	uniforms: StructuredView;
	hasChanged = true;

	constructor(fov: number, aspect: number, near: number, far: number, uniforms: StructuredView) {
		this.fov = fov;
		this.aspect = aspect;
		this.near = near;
		this.far = far;
		this.uniforms = uniforms;
	}

	set(data: any) {
		this.uniforms.set(data);
		this.hasChanged = true;
	}

	recalculateProjectionMatrix() {
		mat4.perspective(this.fov, this.aspect, this.near, this.far, this.uniforms.views.projection);
		this.hasChanged = true;
	}
}

const EPSILON = 0.0001;

enum MouseButtons {
	NONE =		0,
	PRIMARY = 	1 << 0,
	SECONDARY = 1 << 1,
	AUXILIARY = 1 << 2,
	BACK = 		1 << 3,
	FORWARD = 	1 << 4,
}

enum TrackballState {
	NONE =			0,
	INTERPOLATING =	1 << 0,
	ROTATING =		1 << 1,
	ZOOMING =		1 << 2,
	PANNING =		1 << 3,
};

function isFlagSet(v: number, flag: number) { return (v & flag) === flag; }

// Adapted from https://threejs.org/docs/#examples/en/controls/TrackballControls
export default class TrackballControls {
	camera: Camera;
	domElement: HTMLCanvasElement;

	#stateFlags = TrackballState.NONE;

	rotateSpeed = 1;
	zoomSpeed = .05;
	panSpeed = .3;
	dampingFactor = 0.2;

	#lookAt = vec3.fromValues(0, 0, 0);
	#position = vec3.fromValues(0, 0, 0);

	#eye = vec3.fromValues(0, 0, 0);
	#up = vec3.clone(UP);
	
	#currRot = vec3.fromValues(0, 0, 0);
	#targetRot = vec3.fromValues(0, 0, 0);
	#currZoom = 0;
	#targetZoom = 0;
	#currPan = vec2.fromValues(0, 0);
	#targetPan = vec2.fromValues(0, 0);

	// Only used when interpolating between poses
	interpolateSpeed = 0.5;
	#targetPosition = vec3.fromValues(0, 0, 0);
	#currentOrientation = quat.identity();
	#targetOrientation = quat.identity();

	get position() { return this.#position; }

	constructor(camera: Camera, domElement: HTMLCanvasElement, lookAt: Vec3 = vec3.fromValues(0, 0, 0), initialPosition: Vec3 = vec3.fromValues(0, 0, -2)) {
		this.camera = camera;
		this.domElement = domElement;

		this.#lookAt = lookAt;
		this.#position = initialPosition;
		this.#eye = vec3.sub(this.#position, this.#lookAt);

		this.#currZoom = vec3.len(this.#eye);
		this.#targetZoom = this.#currZoom;

		mat4.lookAt(
			this.#position,
			this.#lookAt,
			this.#up,
			this.camera.uniforms.views.view
		);

		this.camera.set({ position: this.#position });

		domElement.oncontextmenu = (e) => e.preventDefault();
		domElement.onwheel = (e) => {
			this.#stateFlags &= ~TrackballState.INTERPOLATING;
			this.#stateFlags |= TrackballState.ZOOMING;
			this.#targetZoom += Math.sign(e.deltaY)
		};
		
		domElement.onmousedown = (e) => {
			if (isFlagSet(e.buttons, MouseButtons.PRIMARY)) {
				this.#stateFlags &= ~TrackballState.INTERPOLATING;
				this.#stateFlags |= TrackballState.PANNING;
				this.#currPan = this.#getMouseOnScreen(e.clientX, e.clientY);
				vec2.clone(this.#currPan, this.#targetPan);

			}
			if (isFlagSet(e.buttons, MouseButtons.SECONDARY)) {
				this.#stateFlags &= ~TrackballState.INTERPOLATING;
				this.#stateFlags |= TrackballState.ROTATING;
				this.#currRot = this.#getMouseProjectedToBall(e.clientX, e.clientY);
				vec3.clone(this.#currRot, this.#targetRot);
			}
		}
			
		domElement.onmousemove = (e) => {
			if (isFlagSet(e.buttons, MouseButtons.PRIMARY)) {
				this.#stateFlags &= ~TrackballState.INTERPOLATING;
				this.#stateFlags |= TrackballState.PANNING;
				this.#targetPan = this.#getMouseOnScreen(e.clientX, e.clientY);

			}
			if (isFlagSet(e.buttons, MouseButtons.SECONDARY)) {
				this.#stateFlags &= ~TrackballState.INTERPOLATING;
				this.#stateFlags |= TrackballState.ROTATING;
				this.#targetRot = this.#getMouseProjectedToBall(e.clientX, e.clientY);
			}
		}		
	}

	update() {
		if (this.#stateFlags === TrackballState.NONE) return;
		
		if (isFlagSet(this.#stateFlags, TrackballState.INTERPOLATING)) {
			if (!this.#interpolatePose()) 
				this.#stateFlags &= ~TrackballState.INTERPOLATING;
		} else {
			this.#eye = vec3.sub(this.#position, this.#lookAt);

			if (isFlagSet(this.#stateFlags, TrackballState.ROTATING) && !this.#rotateCamera())
				this.#stateFlags &= ~TrackballState.ROTATING;
			if (isFlagSet(this.#stateFlags, TrackballState.ZOOMING) && !this.#zoomCamera())
				this.#stateFlags &= ~TrackballState.ZOOMING;
			if (isFlagSet(this.#stateFlags, TrackballState.PANNING) && !this.#panCamera())
				this.#stateFlags &= ~TrackballState.PANNING;

			this.#position = vec3.add(this.#lookAt, this.#eye);
		}

		if (this.#stateFlags !== TrackballState.NONE) {
			mat4.lookAt(
				this.#position,
				this.#lookAt,
				this.#up,
				this.camera.uniforms.views.view
			);

			this.camera.set({ position: this.#position });
		}
	}

	#rotateCamera(): boolean {
		let angle = Math.acos(vec3.dot(this.#currRot, this.#targetRot) / (vec3.len(this.#currRot) * vec3.len(this.#targetRot)));
		if (isNaN(angle) || Math.abs(angle) < EPSILON) {
			vec3.clone(this.#targetRot, this.#currRot);
			return false;
		}

		const axis = vec3.cross(this.#currRot, this.#targetRot);
		vec3.normalize(axis, axis);
		if (isNaN(axis[0]) || isNaN(axis[1]) || isNaN(axis[2])) return false;

		angle *= this.rotateSpeed;

		const rotation = quat.fromAxisAngle(axis, -angle);
		vec3.transformQuat(this.#eye, rotation, this.#eye);
		vec3.transformQuat(this.#up, rotation, this.#up);
		quat.mul(rotation, this.#currentOrientation, this.#currentOrientation);
		vec3.transformQuat(this.#targetRot, rotation, this.#targetRot);

		quat.fromAxisAngle(axis, angle * (this.dampingFactor - 1), rotation);
		vec3.transformQuat(this.#currRot, rotation, this.#currRot);
		return true;
	}

	#zoomCamera(): boolean {
		const factor = 1 + (this.#targetZoom - this.#currZoom) * this.zoomSpeed;

		if (Math.abs(1 - factor) < EPSILON || Math.abs(factor) < EPSILON) {
			this.#currZoom = this.#targetZoom;
			return false;
		}

		vec3.scale(this.#eye, factor, this.#eye);
		this.#currZoom += (this.#targetZoom - this.#currZoom) * this.dampingFactor;
		return true;
	}

	#panCamera(): boolean {
		const delta = vec2.sub(this.#targetPan, this.#currPan);
		if (vec2.len(delta) < EPSILON) {
			vec2.clone(this.#targetPan, this.#currPan);
			return false;
		}
		
		const scaledDelta = vec2.scale(delta, vec3.len(this.#eye) * this.panSpeed);
		const pan = vec3.normalize(vec3.cross(this.#eye, this.#up));
		vec3.scale(pan, scaledDelta[0], pan);
		
		const up = vec3.normalize(this.#up);		
		vec3.addScaled(pan, up, scaledDelta[1], pan);
		
		vec3.add(this.#position, pan, this.#position);
		vec3.add(this.#lookAt, pan, this.#lookAt);
		
		vec2.addScaled(this.#currPan, delta, this.dampingFactor, this.#currPan);
		return true;
	}

	#interpolatePose(): boolean {
		const deltaPos = vec3.sub(this.#targetPosition, this.#position);

		if (vec3.len(deltaPos) < EPSILON) {
			vec3.clone(this.#targetPosition, this.#position);
			quat.clone(this.#targetOrientation, this.#currentOrientation);
			return false;
		}

		const r = vec3.dist(this.position, this.#lookAt);
		const factor = this.interpolateSpeed * this.dampingFactor
		vec3.addScaled(this.#position, deltaPos, factor, this.#position);

		quat.slerp(this.#currentOrientation,
				this.#targetOrientation,
				factor, this.#currentOrientation);

		vec3.transformQuat(UP, this.#currentOrientation, this.#up);
		vec3.transformQuat(vec3.negate(FORWARD), this.#currentOrientation, this.#eye);
		vec3.addScaled(this.#position, vec3.normalize(this.#eye), -r, this.#lookAt);

		return true;
	}

	#getMouseOnScreen(mouseX: number, mouseY: number) {
		return vec2.fromValues(
			mouseX / this.domElement.width,
			mouseY / this.domElement.height
		);
	}

	#getMouseProjectedToBall(mouseX: number, mouseY: number) {
		const mouseOnBall = vec3.fromValues(
			mouseX / (this.domElement.width * 0.5) - 1,
			1 - mouseY / (this.domElement.height * 0.5),
			0.0
		);

		let length = vec3.lenSq(mouseOnBall);
		if (length < Math.SQRT1_2) mouseOnBall[2] = Math.sqrt(1 - length * length);
		else mouseOnBall[2] = 0.5 / length;

		const eye = vec3.normalize(this.#eye);

		const projection = vec3.scale(vec3.normalize(this.#up), mouseOnBall[1]);
		vec3.addScaled(projection, vec3.normalize(vec3.cross(this.#up, eye)), mouseOnBall[0], projection);
		vec3.addScaled(projection, eye, mouseOnBall[2], projection);

		return projection;
	}

	setPose(position: Vec3, rotation: Mat3) {
		vec3.clone(position, this.#targetPosition);
		quat.fromMat(rotation, this.#targetOrientation);
		quat.inverse(this.#targetOrientation, this.#targetOrientation);
		this.#stateFlags = TrackballState.INTERPOLATING;
	}
}