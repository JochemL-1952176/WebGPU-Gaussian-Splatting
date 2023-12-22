import { mat4, vec3, Vec3 } from "wgpu-matrix";

export class OrbitCamera {
	#azimuthAngle: number;
	#polarAngle: number;
	#distance: number;
	
	#position: Vec3;
	#target: Vec3;
	#up: Vec3 = vec3.fromValues(0, 1, 0);

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
		const lookat = vec3.sub(this.#target, this.#position);

		this.#azimuthAngle = Math.atan2(this.#position[0], this.#position[1]),
		this.#polarAngle = Math.atan2(Math.hypot(lookat[0], lookat[1]), lookat[2])
		this.update();

		domElement.oncontextmenu = (e) => e.preventDefault();
		domElement.onwheel = (e) => this.scale(-e.deltaY / 1000);
		domElement.onmousemove = (e) => {
			if (e.buttons == 2) {
				this.rotate((e.movementX / domElement.width) * 10, (e.movementY / domElement.height) * 10);
			} else if (e.buttons == 1) {
				this.pan(e.movementX / domElement.width, e.movementY / domElement.height);
			}
		}
	}

	update() {
		const R = mat4.identity();
		mat4.mul(R, mat4.rotationZ(this.#azimuthAngle), R);
		mat4.mul(R, mat4.rotationX(this.#polarAngle), R);

		this.#position = vec3.add(this.#target, vec3.transformMat4(vec3.fromValues(0, 0, this.#distance), R));
		this.#up = vec3.normalize(vec3.transformMat4(vec3.fromValues(0, 1, 0), R));
		this.#onChange();
	}

	rotate(azimuthDelta: number, polarDelta: number) {
		this.#azimuthAngle -= azimuthDelta;
		this.#polarAngle -= polarDelta;
		this.update();
	}

	scale(amount: number) {
		this.#distance = Math.max(0.1, this.#distance - amount);
		this.update();
	}

	pan(xDir: number, yDir: number) {
		const forward = vec3.normalize(vec3.sub(this.#target, this.#position));
		const right = vec3.cross(forward, this.#up);

		vec3.sub(this.#target, vec3.scale(right, xDir * this.#distance) , this.#target);
		vec3.add(this.#target, vec3.scale(this.#up, yDir * this.#distance) , this.#target);
		this.update();

	}

	getViewMatrix() {
		return mat4.lookAt(this.#position, this.#target, this.#up);
	}
}