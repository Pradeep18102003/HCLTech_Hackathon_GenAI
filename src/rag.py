from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.embedding import VectorStoreManager

class RAGApplication:
    """
    The main application logic that connects the Retriever to the Generator (LLM).
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        
        # 1. Initialize Gemini LLM
        # We use 'gemini-1.5-flash-latest' as it is fast and efficient for RAG.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5
        )

        # 2. Define the Prompt Template
        # This is where we combine the Context and the Question.
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful and knowledgeable AI assistant.
            help with users query

            Context:
            {context}

            User Question: 
            {question}

            Answer:
            """
        )

    def query(self, question: str):
        """
        Retrieves context and generates an answer.
        """
        # A. Retrieve top 5 relevant chunks from ChromaDB
        retriever = self.vector_store.get_retriever(k=5)
        
        # B. Build the LangChain Pipeline
        # 1. Retrieve docs -> 2. Format them -> 3. Pass to Prompt -> 4. Pass to Gemini -> 5. Parse String
        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print(f"User Query: {question}")
        print("Searching for relevant context...")
        
        # C. Execute the chain
        response = chain.invoke(question)
        return response

    @staticmethod
    def _format_docs(docs):
        """Helper to combine multiple retrieved documents into a single text block."""
        return "\n\n".join(doc.page_content for doc in docs)