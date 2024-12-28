# How I Built a Local RAG App for PDF Q&A | Streamlit | LLAMA 3.x | 2025

![RAG Assistant with Chats](local_rag_aasitant_chats.png)

Welcome to my project repository! In this article, I’ll walk you through how I built a **Local RAG (Retrieval-Augmented Generation) App** for **PDF Q&A** using **Streamlit** and the **LLAMA 3.x** model. This app allows users to upload a PDF, convert it into embedding vectors locally, and ask questions about the document. The local LLAMA LLM model provides accurate answers based on the PDF content.

---

## 📌 **Project Overview**

This project is a **local AI-powered PDF chatbot** that leverages the power of **RAG (Retrieval-Augmented Generation)** to answer user questions based on the content of uploaded PDFs. Here’s what makes it special:

- **Local Processing**: Everything runs on your local machine, ensuring privacy and security.
- **Streamlit GUI**: A user-friendly interface for uploading PDFs and interacting with the chatbot.
- **LLAMA 3.x Model**: A powerful local language model for generating accurate responses.
- **Embedding Vectors**: PDFs are converted into embedding vectors for efficient retrieval and analysis.

---

## 🛠️ **Technologies Used**

- **Python**: The core programming language for building the app.
- **Streamlit**: For creating the interactive web interface.
- **LLAMA 3.x**: The local language model used for generating responses.
- **LangChain**: For handling document loading, text splitting, and embeddings.
- **Chroma**: A vector database for storing and retrieving document embeddings.
- **Ollama**: For managing local LLM models and embeddings.

---

## 🚀 **How It Works**

1. **Upload a PDF**: Users can upload a PDF file through the Streamlit interface.
2. **Convert to Embeddings**: The PDF is processed and converted into embedding vectors using **OllamaEmbeddings**.
3. **Ask Questions**: Users can ask questions about the PDF content.
4. **Get Answers**: The LLAMA 3.x model retrieves relevant information from the embeddings and generates accurate answers.

---

## 📂 **Code Structure**

Here’s a breakdown of the key components of the code:

### **1. Streamlit App (`streamlit_app.py`)**
This is the main file that runs the Streamlit application. It handles:
- PDF upload and processing.
- Embedding generation using **OllamaEmbeddings**.
- Question answering using the LLAMA 3.x model.
- Chat interface for user interaction.

```python
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
```
## 2. RAG Pipeline

The app uses a **RAG pipeline** to:

1. **Split the PDF into chunks.**
2. **Generate embeddings** for each chunk.
3. **Retrieve relevant chunks** based on user questions.
4. **Generate answers** using the **LLAMA 3.x model**.

---

### 🖼️ Screenshots

Here are some screenshots of the app in action:

- **RAG Assistant Interface**
![Your PDF Assistant ](https://github.com/seotanvirbd/Local-PDF-RAG-Assistant/blob/main/local_pad_rag_assistant.png)
---

### 🛠️ Installation & Setup

To run this project locally, follow these steps:

#### 1. Clone the Repository:

```bash
git clone https://github.com/yourusername/local-rag-app.git
cd local-rag-app
```
Install Dependencies:


pip install -r requirements.txt
Run the Streamlit App:


streamlit run streamlit_app.py

Upload a PDF and Start Chatting:

Open the app in your browser.

Upload a PDF file.

Ask questions and get answers!



## 🧠 **How the RAG Pipeline Works**

The **Retrieval-Augmented Generation (RAG)** pipeline is the backbone of this project. Here’s a step-by-step breakdown of how it works:

1. **PDF Upload**: The user uploads a PDF file through the Streamlit interface.
2. **Text Extraction**: The PDF is processed using **PyPDFLoader** to extract text content.
3. **Text Splitting**: The extracted text is split into smaller chunks using **RecursiveCharacterTextSplitter**.
4. **Embedding Generation**: Each text chunk is converted into embedding vectors using **OllamaEmbeddings**.
5. **Vector Storage**: The embeddings are stored in a **Chroma** vector database for efficient retrieval.
6. **Question Processing**: When the user asks a question, the app retrieves the most relevant text chunks from the vector database.
7. **Answer Generation**: The **LLAMA 3.x** model generates a response based on the retrieved chunks.

---

## 🎯 **Key Features**

- **Local Processing**: All data processing and model inference happen locally, ensuring privacy and security.
- **User-Friendly Interface**: The Streamlit GUI makes it easy for users to upload PDFs and interact with the chatbot.
- **Customizable Models**: Users can choose from locally available LLM models (e.g., LLAMA 3.x) for generating responses.
- **Multi-Query Retrieval**: The app uses **MultiQueryRetriever** to generate multiple versions of the user’s question, improving retrieval accuracy.
- **PDF Visualization**: Users can view the uploaded PDF pages directly in the app.

---

## 📈 **Future Enhancements**

Here are some ideas for improving the app in the future:
- **Support for Multiple File Formats**: Extend the app to handle DOCX, TXT, and other file formats.
- **Cloud Deployment**: Deploy the app on a cloud platform for remote access.
- **Advanced NLP Features**: Add features like summarization, sentiment analysis, and keyword extraction.
- **User Authentication**: Implement user authentication to secure the app.

---

## 🤝 **Contribution**

If you’d like to contribute to this project, feel free to fork the repository and submit a pull request. Here are some areas where you can help:
- Improve the UI/UX of the Streamlit app.
- Optimize the RAG pipeline for better performance.
- Add support for additional LLM models.

---

## 🙏 **Acknowledgments**

- **Streamlit**: For providing an amazing framework for building interactive web apps.
- **LangChain**: For simplifying the implementation of RAG pipelines.
- **LLAMA 3.x**: For being a powerful and versatile language model.
- **Ollama**: For making it easy to manage local LLM models.

---

## 📜 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🌟 **Star the Repository**

If you found this project useful, don’t forget to ⭐️ star the repository and share it with others. Your support motivates me to keep building and sharing cool projects!

---

## 📞 **Let’s Connect**

I’m always open to new opportunities, collaborations, and discussions. Feel free to reach out via any of the following channels:

- **WhatsApp**: [Chat with Me](https://api.whatsapp.com/send?phone=8801687373830)
- **Facebook**: [Follow Me](https://www.facebook.com/seotanvirbd)
- **LinkedIn**: [Connect with Me](https://www.linkedin.com/in/seotanvirbd/)
- **YouTube**: [Watch My Videos](https://www.youtube.com/@tanvirbinali2200)
- **Email**: [tanvirafra1@gmail.com](mailto:tanvirafra1@gmail.com)
- **Blog & Portfolio**: [Visit My Website](https://seotanvirbd.com/)
- **Upwork**: [Hire Me](https://www.upwork.com/freelancers/~010fc1db7bfe386976?mp_source=share)

---

Thank you for checking out my project! I hope you found it interesting and inspiring. Let’s continue building the future of AI together! 🚀
