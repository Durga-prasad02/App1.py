// Generate uncompressed BMP images for sample waste categories
const fs = require('fs');
const path = require('path');

function createBMP(width, height, drawFn) {
  const rowSize = Math.floor((24 * width + 31) / 32) * 4;
  const pixelArraySize = rowSize * height;
  const fileSize = 54 + pixelArraySize;
  const buf = Buffer.alloc(fileSize);

  // BMP Header
  buf.write('BM', 0); // Signature
  buf.writeUInt32LE(fileSize, 2);
  buf.writeUInt32LE(0, 6); // Reserved
  buf.writeUInt32LE(54, 10); // Offset to pixel data

  // DIB Header (BITMAPINFOHEADER)
  buf.writeUInt32LE(40, 14); // Header size
  buf.writeInt32LE(width, 18);
  buf.writeInt32LE(height, 22); // Bottom to top
  buf.writeUInt16LE(1, 26); // Color planes
  buf.writeUInt16LE(24, 28); // Bits per pixel
  buf.writeUInt32LE(0, 30); // Compression (none)
  buf.writeUInt32LE(pixelArraySize, 34);
  buf.writeInt32LE(2835, 38); // 72 DPI
  buf.writeInt32LE(2835, 42);
  buf.writeUInt32LE(0, 46);
  buf.writeUInt32LE(0, 50);

  // Fill background
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = 54 + (height - 1 - y) * rowSize + x * 3;
      buf[offset] = 245;     // B
      buf[offset + 1] = 245; // G
      buf[offset + 2] = 245; // R
    }
  }

  // Draw custom shapes
  drawFn((x, y, r, g, b) => {
    if (x >= 0 && x < width && y >= 0 && y < height) {
      const offset = 54 + (height - 1 - y) * rowSize + x * 3;
      buf[offset] = b;
      buf[offset + 1] = g;
      buf[offset + 2] = r;
    }
  }, width, height);

  return buf;
}

const baseDir = path.join(__dirname, 'sample_images');

const categories = [
  {
    name: 'Cardboard',
    file: 'cardboard_sample.bmp',
    draw: (setPixel, w, h) => {
      // Brown box
      for (let y = 60; y < 240; y++) {
        for (let x = 50; x < 250; x++) {
          if (x < 55 || x > 245 || y < 65 || y > 235 || (y >= 145 && y <= 155)) {
            setPixel(x, y, 139, 90, 43); // Dark brown outline/tape
          } else {
            setPixel(x, y, 193, 154, 107); // Cardboard brown
          }
        }
      }
    }
  },
  {
    name: 'Glass',
    file: 'glass_bottle_sample.bmp',
    draw: (setPixel, w, h) => {
      // Glass bottle with neck and reflection
      for (let y = 40; y < 260; y++) {
        for (let x = 80; x < 220; x++) {
          const isNeck = (y < 90 && x >= 130 && x <= 170);
          const isBody = (y >= 120 && x >= 90 && x <= 210);
          const isShoulder = (y >= 90 && y < 120 && Math.abs(x - 150) <= (y - 90) * 2 + 20);
          if (isNeck || isBody || isShoulder) {
            if (x === 115 || x === 116) {
              setPixel(x, y, 255, 255, 255); // Specular highlight
            } else if (x === 90 || x === 210 || y === 259) {
              setPixel(x, y, 77, 139, 126); // Outline
            } else {
              setPixel(x, y, 167, 215, 197); // Translucent glass tint
            }
          }
        }
      }
    }
  },
  {
    name: 'Metal',
    file: 'metal_can_sample.bmp',
    draw: (setPixel, w, h) => {
      // Cylindrical aluminum can
      for (let y = 50; y < 250; y++) {
        for (let x = 95; x < 205; x++) {
          if (y >= 110 && y <= 180) {
            setPixel(x, y, 239, 68, 68); // Red brand band
          } else {
            const shade = 140 + Math.floor(Math.sin((x - 95) / 110 * Math.PI) * 60);
            setPixel(x, y, shade, shade, shade); // Metallic gradient
          }
        }
      }
    }
  },
  {
    name: 'Paper',
    file: 'paper_doc_sample.bmp',
    draw: (setPixel, w, h) => {
      // White paper with printed lines
      for (let y = 40; y < 260; y++) {
        for (let x = 70; x < 230; x++) {
          if (x === 70 || x === 229 || y === 40 || y === 259) {
            setPixel(x, y, 148, 163, 184); // Gray border
          } else if (y % 20 === 0 && x > 90 && x < 210 && y > 80 && y < 230) {
            setPixel(x, y, 100, 116, 139); // Text lines
          } else {
            setPixel(x, y, 255, 255, 255); // White sheet
          }
        }
      }
    }
  },
  {
    name: 'Plastic',
    file: 'plastic_bottle_sample.bmp',
    draw: (setPixel, w, h) => {
      // Blue plastic bottle
      for (let y = 40; y < 260; y++) {
        for (let x = 90; x < 210; x++) {
          const isCap = (y < 65 && x >= 135 && x <= 165);
          const isNeck = (y >= 65 && y < 95 && Math.abs(x - 150) <= (y - 65) * 1.5 + 15);
          const isBody = (y >= 95 && x >= 95 && x <= 205);
          if (isCap) {
            setPixel(x, y, 37, 99, 235); // Blue cap
          } else if (isNeck || isBody) {
            if (y >= 140 && y <= 185) {
              setPixel(x, y, 59, 130, 246); // Label
            } else if (x === 110 || x === 111) {
              setPixel(x, y, 255, 255, 255); // Glossy highlight
            } else {
              setPixel(x, y, 147, 197, 253); // Clear plastic blue
            }
          }
        }
      }
    }
  },
  {
    name: 'Organic / Food Waste',
    file: 'organic_peel_sample.bmp',
    draw: (setPixel, w, h) => {
      // Green/yellow organic fruit
      const cx = 150, cy = 160;
      for (let y = 60; y < 260; y++) {
        for (let x = 60; x < 240; x++) {
          const distSq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          if (distSq < 70 * 70) {
            setPixel(x, y, 132, 204, 22); // Lime green
          } else if (y >= 65 && y <= 95 && x >= 146 && x <= 154) {
            setPixel(x, y, 120, 53, 15); // Brown stem
          }
        }
      }
    }
  }
];

categories.forEach(cat => {
  const dir = path.join(baseDir, cat.name.replace('/', '_').trim());
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const bmp = createBMP(300, 300, cat.draw);
  fs.writeFileSync(path.join(dir, cat.file), bmp);
  console.log(`Created sample for ${cat.name} at ${path.join(dir, cat.file)}`);
});
