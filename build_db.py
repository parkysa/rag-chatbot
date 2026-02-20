import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 
BASE = "base"

def load_documents():
    loader = PyPDFDirectoryLoader(BASE)
    documents = loader.load()
    return documents

def split_chunks(documents):
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = documents_splitter.split_documents(documents)

    print(len(chunks))

    return chunks

def vectorize_chunks(chunks):

    embeddings_model = OpenAIEmbeddings()
    texts = []
    vectors = []

    for chunk in chunks:
        texts.append(chunk.page_content)
        
    vectors = embeddings_model.embed_documents(texts)

    df = pd.DataFrame({
        "texto": texts,
        "embedding": vectors
    })

    df.to_parquet("db.parquet", index=False)

    print("Database created")

def create_db():
    documents = load_documents()
    chunks = split_chunks(documents)
    vectorize_chunks(chunks)

create_db()