import { Mat4, mat4, Quat, quat, vec2, Vec2, vec3, Vec3 } from "wgpu-matrix";

function cartesianToSpherical(xyz: Vec3) {
	const r = vec3.length(xyz);
	return {
		thetaPhi: vec2.fromValues(
			Math.acos(xyz[1] / r),
			Math.atan2(xyz[2], xyz[0])
		),
		r: r
	}
}

export class OrbitController {
	#distance: number;
	#rotationQuat: Quat;
	#right: Vec3;
	#up: Vec3 = vec3.fromValues(0, -1, 0);

	#position: Vec3;
	#target: Vec3;
	
	#drag = 8;
	#rotationVelocity = vec2.fromValues(0, 0);
	#panningVelocity = vec2.fromValues(0, 0);
	#zoomVelocity = 0;

	static #zoomSensitivity = 3;
	static #rotateSensitivity = 0.1;
	static #panSensitivity = 0.05;

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

		const fromTarget = vec3.sub(this.#position, this.#target);
		const { thetaPhi, r } = cartesianToSpherical(fromTarget);
		this.#distance = r;
		this.#rotationQuat = quat.fromEuler(thetaPhi[1] + (Math.PI / 2), thetaPhi[0] + (Math.PI / 2), 0, 'xyz');
		const R = mat4.fromQuat(this.#rotationQuat);

		this.#right = vec3.transformMat4(vec3.fromValues(1, 0, 0), R);
		this.#up = vec3.transformMat4(vec3.fromValues(0, -1, 0), R);

		domElement.oncontextmenu = (e) => e.preventDefault();
		domElement.onwheel = (e) => { this.#zoomVelocity -= OrbitController.#zoomSensitivity * Math.sign(e.deltaY); };
		domElement.onmousemove = (e) => {
			switch (e.buttons) {
				case 1: // Left mouse button
					vec2.scale(
						vec2.fromValues(e.movementX, e.movementY),
						OrbitController.#panSensitivity,
						this.#panningVelocity);
					break;
				case 2: // Right mouse button
					vec2.scale(
						vec2.fromValues(e.movementX, e.movementY),
						OrbitController.#rotateSensitivity,
						this.#rotationVelocity);
					break;
			}
		}
	}

	getViewMatrix(target?: Mat4) { return mat4.lookAt(this.#position, this.#target, this.#up, target); }
	update(deltaTime: number) {
		this.#rotate(vec2.scale(this.#rotationVelocity, deltaTime));
		this.#pan(vec2.scale(this.#panningVelocity, deltaTime));
		this.#zoom(this.#zoomVelocity * deltaTime);

		const cmpEpsilon = (v: number) => Math.abs(v) > 0.0001;
		if (this.#rotationVelocity.some(cmpEpsilon) ||
			this.#panningVelocity.some(cmpEpsilon) ||
			cmpEpsilon(this.#zoomVelocity)
		) {
			const R = mat4.fromQuat(this.#rotationQuat);
			vec3.add(this.#target, vec3.transformMat4(vec3.fromValues(0, 0, this.#distance), R), this.position);
			this.#right = vec3.transformMat4(vec3.fromValues(1, 0, 0), R);
			this.#up = vec3.transformMat4(vec3.fromValues(0, -1, 0), R);

			this.#onChange();

			vec3.scale(this.#rotationVelocity, 1 - this.#drag * deltaTime, this.#rotationVelocity);
			vec3.scale(this.#panningVelocity, 1 - this.#drag * deltaTime, this.#panningVelocity);
			this.#zoomVelocity *= 1 - this.#drag * deltaTime;
		} else {
			this.#rotationVelocity.fill(0);
			this.#panningVelocity.fill(0);
			this.#zoomVelocity = 0;
		}
	}

	#pan(delta: Vec2) {
		const pan = vec3.scale(this.#right, delta[0] * this.#distance);
		vec3.addScaled(pan, this.#up, delta[1] * this.#distance, pan);

		vec3.add(this.#target, pan, this.#target);
		vec3.add(this.#position, pan, this.#position);
	}

	#rotate(delta: Vec2) {
		const deltaRotation = quat.fromEuler(delta[1], delta[0], 0, 'xyz');
		quat.multiply(this.#rotationQuat, deltaRotation, this.#rotationQuat);
	}

	#zoom(amount: number) { this.#distance = Math.max(0.1, this.#distance - amount); }
}