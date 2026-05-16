/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  basePath: process.env.NODE_ENV === 'production' ? '/BiBo' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/BiBo/' : '',
};

module.exports = nextConfig;
