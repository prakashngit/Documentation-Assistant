from dotenv import load_dotenv
import os


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

class ReadTheDocsRetriever:
    def __init__(self, collection_name, chroma_persist_directory):
        
        self.history = []
        chroma_vector_store = Chroma(embedding_function = OllamaEmbeddings(model="mxbai-embed-large"),
                            collection_name = collection_name, 
                            persist_directory = chroma_persist_directory)

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        self.retrieval_chain_chroma = create_retrieval_chain(
            retriever=chroma_vector_store.as_retriever(), 
            combine_docs_chain=combine_documents_chain,
        )


        custom_rag_template = """
        Use the following pieces of retrieved context to answer the question at the end. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Use three sentences maximum and keep the answer concise. Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:
        """

        custom_rag_prompt = PromptTemplate.from_template(template=custom_rag_template)
        self.custom_rag_chain = (
            {"context": lambda x: self.format_docs(
                chroma_vector_store.as_retriever().invoke(self.get_history_augmented_query(x["question"]))
            ),
            "question": lambda x: x["question"]}
            | custom_rag_prompt 
            | llm 
        ) 
    
    def get_history_augmented_query(self, query):
        """Augment the query with relevant chat history, and use it for retrieval"""
        if not self.history:
            return query
            
        # Get last few turns of conversation for context
        recent_history = self.history[-2:]  # Consider last 2 turns
        history_str = "\n".join([f"Question: {q}\nAnswer: {a}" for (q, a) in recent_history])
        # Combine current query with chat history
        augmented_query = f"Context: {history_str}\nCurrent question: {query}"
        return augmented_query
    

    def format_docs(self, docs):
        return "\n".join([doc.page_content for doc in docs])
    
    def chat(self, query):
        res = self.retrieval_chain_chroma.invoke(input= {"input": query})
        self.history.append((query, res['answer']))
        return res['answer']
    
    def custom_rag_chat(self, query):
        
        return self.custom_rag_chain.invoke(input= {"question": query}).content
    
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


    print("Responses using the LLM using custom retriever prompt RAG. This time the context used for retrieval is augmented with previous 2 queries and responses\n ", "-"*100, "\n")
    ReadTheDocsRetriever.print_answer(query1, pdf_retriever.custom_rag_chat(query1))
    ReadTheDocsRetriever.print_answer(query2, pdf_retriever.custom_rag_chat(query2))
    ReadTheDocsRetriever.print_answer(query3, pdf_retriever.custom_rag_chat(query3))
    ReadTheDocsRetriever.print_answer(query4, pdf_retriever.custom_rag_chat(query4))


    # Observations and Analysis:
    #
    # Issue identified: Comparing responses with and without augmented query shows significant differences
    # in quality and focus:
    #
    # 1. Without Augmented Query (Original):
    #    - Correctly emphasizes SGX throughout
    #    - Mentions key aspects like "unmodified applications in SGX enclaves"
    #    - Discusses security in terms of SGX protection
    #    - Maintains focus on Gramine's core purpose (SGX adoption)
    #
    # 2. With Augmented Query:
    #    - Almost completely misses SGX
    #    - Focuses too much on implementation details (pipes, IPC)
    #    - Gets stuck on Library OS aspects
    #    - Loses sight of Gramine's main purpose
    #
    # Root Cause Analysis:
    # The augmented query with history is causing the retrieval to:
    # - Over-emphasize implementation details from previous answers
    # - Dilute the core SGX-related content
    # - Create responses that, while technically correct about some aspects, 
    #   miss the main point about SGX
    #
    # Note: Gramine is fundamentally a container framework to simplify SGX adoption
    # for applications while keeping them unmodified. This core concept is being
    # lost in the augmented query responses.
    # A solution is presented in the app.py file, where we ask the llm to first use 
    # the chat history and then use the query to rephrase the question and then use the 
    # rephrased question for retrieval. 