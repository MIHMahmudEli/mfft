/** @type {import('next').NextConfig} */

// Detection API backend. Set NEXT_PUBLIC_API_URL to the production API service;
// defaults to the Hugging Face Spaces production service.
let API_URL = process.env.NEXT_PUBLIC_API_URL;
if (!API_URL || API_URL.includes("onrender.com")) {
  API_URL = "https://mohsinelis-mfft-api.hf.space";
}

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
