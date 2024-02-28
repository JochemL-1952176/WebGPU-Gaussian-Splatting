import { defineConfig } from 'vite'

export default defineConfig({
	build: { target: "esnext" },
	server: {
		open: true,
		// Not strictly necessary, enables higher resolution time data from performance.now()
		// https://developer.mozilla.org/en-US/docs/Web/API/Performance_API/High_precision_timing
		headers: {
			"Cross-Origin-Embedder-Policy": "require-corp",
			"Cross-Origin-Opener-Policy": "same-origin",
		  },
	}

})