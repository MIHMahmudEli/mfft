/** @type {import('next').NextConfig} */

// Detection API backend. Set NEXT_PUBLIC_API_URL to the Hugging Face Space
// URL (e.g. https://mohsinelis-mfft-detector-api.hf.space) for production;
// defaults to the local FastAPI server for development.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
