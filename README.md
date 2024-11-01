# Chat with ReadTheDocs using RAG

A Python application that enables conversational interaction with ReadTheDocs documentation using Retrieval-Augmented Generation (RAG). While demonstrated using Gramine SGX's documentation (included by default), the application can work with any ReadTheDocs content that has been locally scraped. The system features history-aware retrieval, allowing for natural follow-up questions and maintaining conversation context.


This project is part of the exercises for the LangChain- Develop LLM powered applications with LangChain course by Eden Marco, and is available via Udemy [here](https://www.udemy.com/course/langchain/learn/). See Section 6 in the course for Marco's version of the project. The main differences are:
-  I use Ollama instead of OpenAI for the embeddings
-  I use Chroma instead of PineCone for the vector store
-  I have ReadTheDocs of Gramine SGX instead of Langchain for illustration. 

For anyone looking to learn LangChain, I highly recommend Marco's course!

## Features
- ReadTheDocs documentation ingestion and embedding storage
- Context-aware chat interface that maintains conversation history
- Intelligent query reformulation using LangChain's rephrase prompts
- Source attribution for answers
- Batched document processing with progress tracking

## Technologies Used
- Python 3.x
- LangChain
- OpenAI GPT models for query answering
- Ollama (local) for embeddings
- ChromaDB for vector storage
- Streamlit for the chat interface

## Two versions of the chat interface:
  - `app.py`: Basic chat functionality
  - `enhanced_app.py`: Enhanced version with additional styling and UI improvements (created using Cursor AI's composer functionality)

## Setup
1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

3. Run either version of the chat interface from the `python` directory:
```
streamlit run app.py
```
or
```
streamlit run enhanced_app.py
```

4. (Optional) To chat with different ReadTheDocs documentation:
   - Scrape and store your chosen ReadTheDocs content locally
   - Update the documentation path in the configuration

## License
This project is licensed under the Apache License, Version 2.0 (APL 2.0). License. See the [LICENSE](LICENSE) file for details.
