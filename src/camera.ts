import { mat4, vec2, Vec2, vec3, Vec3 } from "wgpu-matrix";

export class OrbitCamera {
	#distance: number;
	#rotation: Vec2;
	#forward: Vec3;
	#right: Vec3;
	#up: Vec3 = vec3.fromValues(0, 1, 0);

	#position: Vec3;
	#target: Vec3;
	
	#dragCoefficient: number = 0.95;
	#rotationVelocity: Vec2 = vec2.fromValues(0, 0);
	#panningVelocity: Vec2 = vec2.fromValues(0, 0);
	#zoomVelocity: number = 0;

	static #zoomSensitivity: number = 0.1;
	static #rotateSensitivity: number = 0.003;
	static #panSensitivity: number = 0.0008;

	get position(): Vec3 {
		return this.#position;
	}

	#onChange: () => void = () => {};
	set onChange(func: () => void) {
		this.#onChange = func;
		this.#onChange();
	}

	constructor(position: Vec3, target: Vec3, domElement: HTMLCanvasElement) {
		this.#position = position;
		this.#target = target;

		this.#distance = vec3.dist(this.#position, this.#target);
		this.#forward = vec3.normalize(vec3.sub(this.#target, this.#position));
		const fromTarget = vec3.sub(this.#position, this.#target)

		this.#rotation = vec2.fromValues(
			Math.atan2(fromTarget[1], fromTarget[0]),
			Math.acos(fromTarget[2] / this.#distance)
		);
		
		const R = mat4.identity();
		mat4.mul(R, mat4.rotationZ(Math.PI / 2 + this.#rotation[0]), R);
		mat4.mul(R, mat4.rotationX(this.#rotation[1]), R);
		vec3.normalize(vec3.transformMat4(vec3.fromValues(0, 1, 0), R), this.#up);
		this.#right = vec3.cross(this.#forward, this.#up);

		domElement.oncontextmenu = (e) => e.preventDefault();
		domElement.onwheel = (e) => { this.#zoomVelocity = -OrbitCamera.#zoomSensitivity * Math.sign(e.deltaY); } ;
		domElement.onmousemove = (e) => {
			switch (e.buttons) {
				case 1: // Left mouse button
				this.#panningVelocity = vec2.fromValues(
					OrbitCamera.#panSensitivity * e.movementX,
					OrbitCamera.#panSensitivity * e.movementY);
					break;
				case 2: // Right mouse button
					this.#rotationVelocity = vec2.fromValues(
						OrbitCamera.#rotateSensitivity * e.movementX,
						OrbitCamera.#rotateSensitivity * e.movementY);
					break;
			}
		}
	}

	getViewMatrix() {
		return mat4.lookAt(this.#position, this.#target, this.#up);
	}

	update() {
		this.#rotate(this.#rotationVelocity);
		this.#pan(this.#panningVelocity);
		this.#zoom(this.#zoomVelocity);

		if (Math.abs(this.#rotationVelocity[0]) > 0.0001 ||
			Math.abs(this.#rotationVelocity[1]) > 0.0001 ||
			Math.abs(this.#panningVelocity[0]) > 0.0001 ||
			Math.abs(this.#panningVelocity[1]) > 0.0001 ||
			Math.abs(this.#zoomVelocity) > 0.0001
		) {
			const R = mat4.identity();
			mat4.mul(R, mat4.rotationZ(Math.PI / 2 + this.#rotation[0]), R);
			mat4.mul(R, mat4.rotationX(this.#rotation[1]), R);

			vec3.add(this.#target, vec3.transformMat4(vec3.fromValues(0, 0,  this.#distance), R), this.#position);
			vec3.normalize(vec3.sub(this.#target, this.#position), this.#forward);
			vec3.normalize(vec3.transformMat4(vec3.fromValues(0, 1, 0), R), this.#up);
			vec3.cross(this.#forward, this.#up, this.#right);
			
			this.#onChange();

			vec3.scale(this.#rotationVelocity, this.#dragCoefficient, this.#rotationVelocity);
			vec3.scale(this.#panningVelocity, this.#dragCoefficient, this.#panningVelocity);
			this.#zoomVelocity *= this.#dragCoefficient;
		}
	}

	#rotate(delta: Vec2) {
		vec2.sub(this.#rotation, delta, this.#rotation);
	}

	#zoom(amount: number) {
		this.#distance = Math.max(0.1, this.#distance - amount);
	}

	#pan(delta: Vec2) {
		const pan = vec3.scale(this.#right, -delta[0] * this.#distance);
		vec3.addScaled(pan, this.#up, delta[1] * this.#distance, pan);

		vec3.add(this.#target, pan, this.#target);
		vec3.add(this.#position, pan, this.#position);
	}
}