/** @type {import('next').NextConfig} */

// Detection API backend. Set NEXT_PUBLIC_API_URL to the production API service;
// defaults to the Render FastAPI production service.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://mfft-detection-api.onrender.com";

const nextConfig = {
  output: "standalone",
  images: {
    domains: [],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
