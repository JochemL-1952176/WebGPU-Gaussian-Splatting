import { defineConfig } from 'vite'

export default defineConfig({
	build: {
		target: "esnext"
	},
	server: {
		open: true
	},
	assetsInclude: ["**/*.wgsl"]
})