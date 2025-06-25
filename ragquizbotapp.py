# Install missing packages
#%pip install gradio
#%pip install PyPDF2
#%pip install streamlit

from langchain_community.llms import ollama
import gradio as gr
import numpy as np
import random 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
# Removed import for OllamaChat as it does not exist in langchain_community.chat_model
from dotenv import load_dotenv, find_dotenv 
from langchain import LLMChain, PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceHubEmbeddings
import faiss
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile

llm = ollama.Ollama(model="llama3") 

def load_split_chunk(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

def embed_chunks(chunks):
    texts = [doc.page_content for doc in chunks]
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def generate_mcq(content):
    """
    Generates a multiple-choice question and the correct answer based on the provided content.
    """
    response_schemas = [
        ResponseSchema(name="question", description="The multiple-choice question"),
        ResponseSchema(name="option_a", description="Option A for the question"),
        ResponseSchema(name="option_b", description="Option B for the question"),
        ResponseSchema(name="option_c", description="Option C for the question"),
        ResponseSchema(name="option_d", description="Option D for the question"),
        ResponseSchema(name="correct_answer", description="The correct answer (should match the text of one of the options)"),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
Generate a multiple-choice question based on the following content.

Content: {content}

{format_instructions}
""",
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    result = chain.run(content)
    result = output_parser.parse(result)

    question_text = result.get("question")
    answer_options = {
        "A": result.get("option_a"),
        "B": result.get("option_b"),
        "C": result.get("option_c"),
        "D": result.get("option_d"),
    }

    correct_answer_text = result.get("correct_answer")

    # Ensure consistent order before shuffling
    ordered_options = list(answer_options.items())
    random.shuffle(ordered_options)

    # Re-map after shuffle
    new_labels = ['A', 'B', 'C', 'D']
    shuffled_options = {new_label: text for new_label, (_, text) in zip(new_labels, ordered_options)}

    # Find new correct label
    correct_label = next(label for label, text in shuffled_options.items() if text == correct_answer_text)

    # Format question
    formatted_question = question_text + '\n' + "\n".join([f"{label}) {text}" for label, text in shuffled_options.items()])

    explanation = generate_explanation(question_text, correct_answer_text)

    return formatted_question, list(shuffled_options.keys()), correct_label, explanation


def generate_explanation(question, correct_answer):
    """
    Generates an explanation for a question and answer.
    """
    prompt = PromptTemplate(
        input_variables=["question", "correct_answer"],
        template="Provide an explanation for the following question and answer:\n\nQuestion: {question}\nCorrect Answer: {correct_answer}\n\nExplanation:"
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run({"question": question, "correct_answer": correct_answer})


def generate_question(content):
    """
    Entry function to generate a full MCQ and explanation from content.
    """
    return generate_question_and_answer(content)

st.title("📚 Quiz Question Generator (RAG + LangChain)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        chunks = load_split_chunk(uploaded_file)
        retriever = embed_chunks(chunks)

    query = st.text_input("Ask a topic/question to generate a quiz from:")
    
    if query:
        docs = retriever.get_relevant_documents(query, k=1)
        context = docs[0].page_content

        with st.spinner("Generating question..."):
            question_text, correct_label, correct_answer = generate_mcq(context)
            explanation = generate_explanation(question_text, correct_answer)

        st.markdown(f"### 🧠 Question:\n{question_text}")
        st.markdown(f"**✅ Correct Answer:** {correct_label}) {correct_answer}")
        st.markdown(f"**💡 Explanation:** {explanation}")