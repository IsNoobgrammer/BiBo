'use client';

import { useState } from 'react';

interface PlotImageProps {
  src: string;
  alt: string;
  className?: string;
}

export function PlotImage({ src, alt, className = '' }: PlotImageProps) {
  const [zoomed, setZoomed] = useState(false);
  const basePath = process.env.NODE_ENV === 'production' ? '/BiBo' : '';
  const fullSrc = `${basePath}/plots/${src}.png`;

  return (
    <>
      <img
        src={fullSrc}
        alt={alt}
        loading="lazy"
        onClick={() => setZoomed(true)}
        className={`w-full rounded-lg cursor-zoom-in transition-opacity duration-200 hover:opacity-90 ${className}`}
      />
      {zoomed && (
        <div className="img-overlay" onClick={() => setZoomed(false)}>
          <img
            src={fullSrc}
            alt={alt}
            className="max-w-[95%] max-h-[95%] rounded-lg shadow-2xl"
          />
        </div>
      )}
    </>
  );
}
