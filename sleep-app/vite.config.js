import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'Dự đoán chất lượng giấc ngủ',
        short_name: 'SleepAI',
        description: 'Phân tích chất lượng giấc ngủ bằng AI',
        theme_color: '#0a0614',
        background_color: '#0a0614',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png' }
        
        ]
      }
    })
  ]
})