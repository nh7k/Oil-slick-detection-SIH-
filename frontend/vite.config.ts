import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies /api to the FastAPI backend. Override the proxy target with
// VITE_PROXY_TARGET, or bypass the proxy entirely by setting VITE_API_URL
// (an absolute base URL used by the browser, e.g. https://api.example.org).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000';
  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': { target, changeOrigin: true },
      },
    },
    preview: {
      port: 4173,
      proxy: {
        '/api': { target, changeOrigin: true },
      },
    },
    build: {
      sourcemap: true,
      chunkSizeWarningLimit: 1200,
      rollupOptions: {
        output: {
          manualChunks: { maplibre: ['maplibre-gl'], react: ['react', 'react-dom', 'react-router-dom'] },
        },
      },
    },
  };
});
