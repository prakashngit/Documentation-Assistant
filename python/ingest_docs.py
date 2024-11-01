from dotenv import load_dotenv

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm.auto import tqdm

import os 
   
def ingest_readthedocs(docs_root, docs_folder, collection_name = "default_collection"):
    """ Given a folder containing a ReadTheDocs site, 
    ingest the data into a local Chroma vector store. 
    Embeddings are generated using Ollama deployed locally.

    Pass in a name to create a new collection, or leave blank to add to the default collection.
    The vector store is persisted locally in ./chroma_db
    
    Returns the number of chunks embedded, the name of the collection, and the path to the local vector store.
    """
    
    # load the docs
    docs_path = os.path.join(docs_root, docs_folder)
    loader = ReadTheDocsLoader(docs_path)
    raw_documents = loader.load()
    
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    
    # update the source metadata of the documents to be the relative path from the docs_root
    # so that the source metadata is the actual url of the document
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace(docs_root, "https://")
        doc.metadata.update({"source": new_url})

    # embed the chunks
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_persist_directory = os.path.join(script_dir, "chroma_db")
    
    # Create the vector store
    vector_store = Chroma(collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Process documents in batches with progress bar
    batch_size = 50
    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)

    for i in tqdm(range(0, len(documents), batch_size), total=total_batches, desc="Adding documents"):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(documents=batch)

    chroma_client = vector_store._client
    collection = chroma_client.get_collection(collection_name)
    return collection.count(), collection_name, chroma_persist_directory

if __name__ == "__main__":
    print("Ingesting data...")
    load_dotenv()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_root = os.path.join(script_dir, "../gramine-docs/")
    docs_folder = 'gramine.readthedocs.io/en/stable/'
    
    count, name, chroma_persist_directory = ingest_readthedocs(docs_root, docs_folder, collection_name="gramine_docs")
    print(f"Ingested {count} chunks into collection {name}")
    print(f"Chroma vector store persisted in {chroma_persist_directory}")
           