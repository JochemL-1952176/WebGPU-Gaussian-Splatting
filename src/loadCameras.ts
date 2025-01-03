import { Mat3, Vec3, mat3 } from "wgpu-matrix"

export type CameraData = {
	id: number,
	img_name: string,
	width: number,
	height: number,
	position: Vec3,
	rotation: Mat3,
	fy: number,
	fx: number
}

export default async function loadCameras(camerasFile: Blob): Promise<Array<CameraData>> {
	const cameras: Array<CameraData> = JSON.parse(await camerasFile.text())
	cameras.forEach((cam: CameraData) => {
		mat3.set(
			cam.rotation[0][0], cam.rotation[0][1], cam.rotation[0][2],
			cam.rotation[1][0], cam.rotation[1][1], cam.rotation[1][2],
			cam.rotation[2][0], cam.rotation[2][1], cam.rotation[2][2],
			cam.rotation
		);
	});

	return cameras;
}