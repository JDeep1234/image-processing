# Smart OCR Product Analyzer

A modern web application that uses OCR (Optical Character Recognition) to extract product details from images. Built with React, TypeScript, and Python, this application leverages machine learning to automatically detect and extract information such as brand name, pack size, expiry date, batch number, and MRP from product images.

## Features

- 🖼️ Drag-and-drop image upload interface
- 🔍 Automatic extraction of product details
- ⚡ Real-time analysis feedback
- 📊 Clean tabular display of results
- 🚨 Expiry date validation and warnings
- 📱 Responsive design for all devices
- 🎯 High accuracy text detection
- 🔒 Error handling and validation

## Tech Stack

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- Vite
- Axios
- React Dropzone
- Lucide React (icons)
- date-fns
- Zod (runtime type checking)

### Backend
- Flask
- PyTorch
- Transformers (Hugging Face)
- PIL (Python Imaging Library)
- CUDA support for GPU acceleration
- Docker

## Prerequisites

- Node.js 18+
- Python 3.8+
- Docker (for deployment)
- CUDA-compatible GPU (optional, for faster processing)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-ocr-product-analyzer.git
cd smart-ocr-product-analyzer
```

2. Install frontend dependencies:
```bash
npm install
```

3. Create and configure environment variables:
```bash
cp .env.example .env
```
Update the `VITE_API_URL` in `.env` with your backend API URL.

4. Start the frontend development server:
```bash
npm run dev
```

5. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Development

### Frontend Structure
```
src/
├── components/     # React components
├── hooks/         # Custom React hooks
├── services/      # API services
├── types/         # TypeScript type definitions
├── utils/         # Utility functions
└── config/        # Environment configuration
```

### Backend Structure
```
backend/
├── app.py         # Main Flask application
├── Dockerfile     # Container configuration
└── requirements.txt
```

## Deployment

### Frontend
1. Build the production bundle:
```bash
npm run build
```

2. Deploy the `dist` folder to your preferred hosting service.

### Backend
1. Build the Docker image:
```bash
cd backend
docker build -t smart-ocr-analyzer .
```

2. Run the container:
```bash
docker run -p 8080:8080 smart-ocr-analyzer
```

For production deployment, consider using AWS SageMaker for the ML model deployment and container orchestration services like AWS ECS or Kubernetes.

## Environment Variables

### Frontend
- `VITE_API_URL`: Backend API endpoint URL

### Backend
- `PORT`: Server port (default: 8080)
- `FLASK_ENV`: Environment mode (development/production)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Microsoft GIT Base Model](https://huggingface.co/microsoft/git-base-textcaps) for OCR capabilities
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for ML model integration
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [React Dropzone](https://react-dropzone.js.org/) for file upload functionality
