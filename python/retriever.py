from dotenv import load_dotenv
import os


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from typing import List, Dict, Any, Optional

class ReadTheDocsRetriever:
    def __init__(self, collection_name, chroma_persist_directory):
        
        chroma_vector_store = Chroma(embedding_function = OllamaEmbeddings(model="mxbai-embed-large"),
                            collection_name = collection_name, 
                            persist_directory = chroma_persist_directory,
                            collection_metadata={"hnsw:space": "cosine"})

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        history_aware_retriever = create_history_aware_retriever(llm, chroma_vector_store.as_retriever(), rephrase_prompt)
        
        self.retrieval_chain_chroma = create_retrieval_chain(
            retriever=history_aware_retriever, 
            combine_docs_chain=combine_documents_chain,
        )
       
        
    
    def chat(self, query, return_context=False, chat_history=[]):
        res = self.retrieval_chain_chroma.invoke(input= {"input": query, "chat_history": chat_history})
        if return_context:
            return res['answer'], res['context']
        else:
            return res['answer']
    
    
    @staticmethod
    def print_answer(query, answer):
        print("Question: ", query)
        print("Answer: ", answer)
        print("-"*100, "\n")
    

if __name__ == "__main__":
    print("Retrieving...")
    load_dotenv()
    
    print("Responses using RAG \n ", "-"*100, "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_retriever = ReadTheDocsRetriever(collection_name="gramine_docs", chroma_persist_directory=os.path.join(script_dir, "./chroma_db"))

    query1 = "What is Gramine all about?"
    ReadTheDocsRetriever.print_answer(query1, pdf_retriever.chat(query1))

    query2 = "How does Gramine simplify adoption of secure enclaves?"
    ReadTheDocsRetriever.print_answer(query2, pdf_retriever.chat(query2))

    query3 = "What is the programming model of Gramine?"
    ReadTheDocsRetriever.print_answer(query3, pdf_retriever.chat(query3 ))

    query4 = "What is the security model of Gramine?"
    ReadTheDocsRetriever.print_answer(query4, pdf_retriever.chat(query4))

    