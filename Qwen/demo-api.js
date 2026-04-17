import fs from 'fs';

// ===== ENCODERS =====

async function encodeToBase64(path, mime) {
  const buffer = await fs.promises.readFile(path);
  const base64 = buffer.toString('base64');
  return `data:${mime};base64,${base64}`;
}

// ===== MAIN =====

async function run() {
  const API_KEY_REF = "YOUR_API_KEY";

  // Paths
  const videoPath = 'path/to/video.mp4';
  const pdfPath = 'path/to/document.pdf';
  const imagePath = 'path/to/image.jpg';

  // Encode
  const base64Video = await encodeToBase64(videoPath, 'video/mp4');
  const base64PDF = await encodeToBase64(pdfPath, 'application/pdf');
  const base64Image = await encodeToBase64(imagePath, 'image/jpeg');

  // Request
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${API_KEY_REF}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'google/gemini-2.5-flash',

      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze all inputs (video, image, PDF) and give a combined summary.',
            },

            // PDF
            {
              type: 'file',
              file: {
                filename: 'document.pdf',
                file_data: base64PDF,
              },
            },

            // IMAGE
            {
              type: 'image_url',
              image_url: {
                url: base64Image,
              },
            },

            // VIDEO
            {
              type: 'video_url',
              video_url: {
                url: base64Video,
              },
            },
          ],
        },
      ],

      plugins: [
        {
          id: 'file-parser',
          pdf: {
            engine: 'mistral-ocr',
          },
        },
      ],
    }),
  });

  const data = await response.json();
  console.log(data);
}

run();
