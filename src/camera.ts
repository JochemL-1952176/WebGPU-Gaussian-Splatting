import { mat4, vec2, Vec2, vec3, Vec3 } from "wgpu-matrix";

function sphericalToCartesian(thetaPhi: Vec2, r: number) {
	return vec3.fromValues(
		r * Math.sin(thetaPhi[0]) * Math.cos(thetaPhi[1]),
		r * Math.cos(thetaPhi[0]),
		r * Math.sin(thetaPhi[0]) * Math.sin(thetaPhi[1])
	)
}

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

export class OrbitCamera {
	#distance: number;
	#rotation: Vec2;
	#forward: Vec3;
	#right: Vec3;
	#up: Vec3 = vec3.fromValues(0, -1, 0);

	#position: Vec3;
	#target: Vec3;
	
	#dragCoefficient: number = 0.95;
	#rotationVelocity: Vec2 = vec2.fromValues(0, 0);
	#panningVelocity: Vec2 = vec2.fromValues(0, 0);
	#zoomVelocity: number = 0;

	static #zoomSensitivity: number = 0.025;
	static #rotateSensitivity: number = 0.002;
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

		const fromTarget = vec3.sub(this.#position, this.#target);

		let { thetaPhi, r } = cartesianToSpherical(fromTarget)
		this.#rotation = thetaPhi, this.#distance = r;

		this.#forward = vec3.normalize(vec3.sub(this.#target, this.#position));
		this.#right = vec3.cross(this.#forward, this.#up);

		domElement.oncontextmenu = (e) => e.preventDefault();
		domElement.onwheel = (e) => { this.#zoomVelocity -= OrbitCamera.#zoomSensitivity * Math.sign(e.deltaY); } ;
		domElement.onmousemove = (e) => {
			switch (e.buttons) {
				case 1: // Left mouse button


					vec2.scale(
						vec2.fromValues(e.movementX, e.movementY),
						OrbitCamera.#panSensitivity,
						this.#panningVelocity);
					break;
				case 2: // Right mouse button
					vec2.scale(
						vec2.fromValues(-e.movementY, e.movementX),
						OrbitCamera.#rotateSensitivity,
						this.#rotationVelocity);
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

		this.#rotation[0] = Math.max(0.001, Math.min(Math.PI - 0.001, this.#rotation[0] % 180));
		this.#rotation[1] = this.#rotation[1] % 360;

		const cmpEpsilon = (v: number) => Math.abs(v) > 0.0001;
		if (this.#rotationVelocity.some(cmpEpsilon) ||
			this.#panningVelocity.some(cmpEpsilon) ||
			cmpEpsilon(this.#zoomVelocity)
		) {
			vec3.add(this.#target, sphericalToCartesian(this.#rotation, this.#distance), this.position);
			vec3.normalize(vec3.sub(this.#target, this.#position), this.#forward);
			vec3.cross(this.#forward, this.#up, this.#right);
			
			this.#onChange();

			vec3.scale(this.#rotationVelocity, this.#dragCoefficient, this.#rotationVelocity);
			vec3.scale(this.#panningVelocity, this.#dragCoefficient, this.#panningVelocity);
			this.#zoomVelocity *= this.#dragCoefficient;
		} else {
			this.#rotationVelocity.fill(0);
			this.#panningVelocity.fill(0);
			this.#zoomVelocity = 0;
		}
	}

	#rotate(delta: Vec2) { vec2.sub(this.#rotation, delta, this.#rotation); }
	#zoom(amount: number) { this.#distance = Math.max(0.1, this.#distance - amount); }

	#pan(delta: Vec2) {
		const pan = vec3.scale(this.#right, -delta[0] * this.#distance);
		vec3.addScaled(pan, vec3.cross(this.#right, this.#forward), delta[1] * this.#distance, pan);

		vec3.add(this.#target, pan, this.#target);
		vec3.add(this.#position, pan, this.#position);
	}
}