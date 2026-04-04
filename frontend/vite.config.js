import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/register': 'http://localhost:8000',
      '/login': 'http://localhost:8000',
      '/logout': 'http://localhost:8000',
      '/upload': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/ask': 'http://localhost:8000',
    },
  },
})
