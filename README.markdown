# Kimi-K2 + Tesseract RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) application that enables users to upload PDF or image files, extract text using Tesseract OCR for images and embedded images in PDFs, and query the content using the Kimi-K2-Instruct model from Moonshot AI via HuggingFace's InferenceClient. The app leverages FAISS for vector-based similarity search to provide context-aware responses.
<img width="1600" height="863" alt="image" src="https://github.com/user-attachments/assets/14122b30-0dad-45eb-9b77-10315b0b750a" />

<img width="1600" height="794" alt="image" src="https://github.com/user-attachments/assets/fbdbf435-232f-44df-82d1-e85acd9676cc" />

<img width="1600" height="789" alt="image" src="https://github.com/user-attachments/assets/05863cb2-77b1-4c36-aa23-318d8a5126f5" />

## Features
- Upload and process multiple PDFs or images (PNG, JPG, JPEG).
- Extract text from PDFs using PyMuPDF (fitz) and perform OCR on images using Tesseract.
- Index extracted text into a FAISS vector store for efficient retrieval.
- Query the indexed content with natural language questions, answered by the Kimi-K2-Instruct model.
- User-friendly Streamlit interface with chat-based interaction.

## Prerequisites
- Python 3.8+
- Tesseract-OCR installed on your system (update `pytesseract.pytesseract.tesseract_cmd` in the code to match your Tesseract installation path).
- A HuggingFace API token with access to the Kimi-K2-Instruct model (set as `HF_API_KEY` in the code).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract-OCR:
   - **Windows**: Download and install from [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki). Update the `tesseract_cmd` path in the code if needed.
   - **Linux/Mac**: Install via package manager (e.g., `sudo apt install tesseract-ocr` for Ubuntu or `brew install tesseract` for macOS).

5. Set your HuggingFace API token:
   - Replace `HF_API_KEY` in the code with your HuggingFace API token or set it as an environment variable:
     ```bash
     export HF_TOKEN="your-huggingface-api-token"
     ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the provided local URL (e.g., `http://localhost:8501`) in your browser.

3. Upload PDF or image files via the interface.

4. Click "Process files" to extract text, perform OCR, and index the content.

5. Ask questions in the chat input to query the processed documents. The app retrieves relevant content and generates responses using the Kimi-K2-Instruct model.

## Dependencies
Listed in `requirements.txt`:
- streamlit
- PyMuPDF (fitz)
- pytesseract
- Pillow
- langchain-community
- faiss-cpu
- huggingface_hub

Install them using:
```bash
pip install streamlit PyMuPDF pytesseract Pillow langchain-community faiss-cpu huggingface_hub
```

## Notes
- Ensure Tesseract-OCR is correctly installed and its path is set in the script.
- The app uses the `all-MiniLM-L6-v2` model for embeddings, which is lightweight and effective for text similarity tasks.
- The Kimi-K2-Instruct model requires a valid HuggingFace API token with access to the Moonshot AI provider.
- For large PDFs or images, processing time may vary depending on system resources and file complexity.

## Limitations
- OCR accuracy depends on image quality and Tesseract’s performance.
- The app assumes English text for OCR (`lang="eng"`). Modify the `ocr_bytes` function for other languages.
- The Kimi-K2-Instruct model may occasionally be unavailable due to API constraints.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## Acknowledgments
- Powered by [Streamlit](https://streamlit.io/), [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract), and [HuggingFace](https://huggingface.co/).
- Uses the Kimi-K2-Instruct model by Moonshot AI for natural language processing.
