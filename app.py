
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

# from langchain_ollama import (
#     ChatOllama,
#     OllamaEmbeddings
# )

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

load_dotenv(override=True)

# --------------------------------------------------
# 1. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Knowledge Agent",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Knowledge Agent")
st.caption("Ask questions about your PDF using a local Agentic AI.")


# --------------------------------------------------
# 2. Initialize local AI models
# --------------------------------------------------

@st.cache_resource
def load_models():

    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return llm, embeddings


# --------------------------------------------------
# 3. Process PDF and create vector database
# --------------------------------------------------

@st.cache_resource
def create_vector_database(pdf_bytes):

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp_file:

        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    try:
        # Extract text from PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    finally:
        Path(pdf_path).unlink(missing_ok=True)

    if not documents:
        raise ValueError("No readable text found in PDF.")

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("PDF does not contain extractable text.")

    # Create embeddings and vector database
    _, embeddings = load_models()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_knowledge"
    )

    return vector_db


# --------------------------------------------------
# 4. Create the Agent
# --------------------------------------------------

def create_pdf_agent(vector_db):

    # Define PDF search as an agent tool
    @tool
    def search_pdf(question: str) -> str:
        """
        Search the uploaded PDF for information
        relevant to the user's question.

        Use this tool whenever the answer should
        be based on the uploaded document.
        """

        results = vector_db.similarity_search(
            question,
            k=4
        )

        if not results:
            return "No relevant information found in PDF."

        formatted_results = []

        for document in results:

            page_number = (
                document.metadata.get("page", 0) + 1
            )

            content = document.page_content

            formatted_results.append(
                f"Page {page_number}:\n{content}"
            )

        return "\n\n".join(formatted_results)

    # Initialize local LLM
    llm, _ = load_models()

    # Agent system instructions
    system_prompt = SystemMessage(
        content="""
        You are a helpful PDF Knowledge Agent.

        Your job is to answer questions using
        the uploaded PDF document.

        Instructions:
        1. Use the search_pdf tool to retrieve
           relevant information.
        2. Do not invent information.
        3. If the PDF does not contain the answer,
           clearly say that the information was
           not found in the document.
        4. Explain answers in simple language.
        5. Mention the PDF page numbers used
           in your answer.
        6. You may summarize or explain retrieved
           information, but stay grounded in it.
        """
    )

    # Create a tool-calling ReAct agent
    # agent = create_react_agent(
    #     model=llm,
    #     tools=[search_pdf],
    #     prompt=system_prompt,
    #     debug = True
    # )
    agent = create_agent(
        model=llm,
        tools=[search_pdf],
        system_prompt="""
        You are a PDF Knowledge Agent.

        Use search_pdf to answer questions
        from the uploaded document.

        Do not invent information.
        Mention relevant PDF page numbers.
        """
)

    return agent


# --------------------------------------------------
# 5. PDF Upload
# --------------------------------------------------

uploaded_pdf = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_pdf:

    pdf_bytes = uploaded_pdf.getvalue()

    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    st.success(
        f"Uploaded: {uploaded_pdf.name}"
    )

    try:

        with st.spinner(
            "Reading PDF and creating knowledge base..."
        ):

            vector_db = create_vector_database(
                pdf_bytes
            )

            agent = create_pdf_agent(
                vector_db
            )

        st.success(
            "PDF processed successfully. "
            "You can now ask questions."
        )

    except Exception as error:

        st.error(
            f"Could not process PDF: {error}"
        )

        st.stop()


    # --------------------------------------------------
    # 6. Chat History
    # --------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if (
        "current_pdf_hash" not in st.session_state
        or st.session_state.current_pdf_hash != pdf_hash
    ):

        st.session_state.messages = []
        st.session_state.current_pdf_hash = pdf_hash


    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # --------------------------------------------------
    # 7. User Question
    # --------------------------------------------------

    question = st.chat_input(
        "Ask something about your PDF..."
    )


    if question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        # Execute the agent
        with st.chat_message("assistant"):

            with st.spinner(
                "Agent is searching the PDF..."
            ):

                try:

                    # response = agent.invoke(
                    #     {
                    #         "messages": [
                    #             HumanMessage(
                    #                 content=question
                    #             )
                    #         ]
                    #     }
                    # )

                    response = agent.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=question
                                )
                            ]
                        },
                        config={
                            "tags": [
                                "pdf-chatbot",
                                "local-development"
                            ],
                            "metadata": {
                                "application": "PDF Knowledge Agent",
                                "model": "llama3.2",
                                "vector_database": "ChromaDB",
                                "document_type": "PDF"
                            }
                        }
                    )

                    answer = response[
                        "messages"
                    ][-1].content

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as error:

                    st.error(
                        f"Agent error: {error}"
                    )

else:

    st.info(
        "Please upload a PDF to start chatting."
    )
