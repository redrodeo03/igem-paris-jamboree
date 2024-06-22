from langchain.embeddings import CohereEmbeddings
from faiss import write_index, read_index
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ApifyDatasetLoader
import os
from langchain.document_loaders.base import Document
from langchain.vectorstores import FAISS
from decouple import config
from langchain.document_loaders.csv_loader import CSVLoader


os.environ["COHERE_API_KEY"] = "3Kcl5JLbJPfliBP1zGQLQ1SreitV2ulm9IG9SO3x"
database_id = "QmwTQRFpZYgIHsRxP"

embeddings = CohereEmbeddings()

batch_size = 400
embedding_model = "cl100k_base"
tokenizer = tiktoken.get_encoding('cl100k_base')

loader = CSVLoader(file_path='./final.csv', encoding="utf-8")

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""])


chunks = text_splitter.split_documents(data) 

str_list = []

for i in range(len(chunks)):
    str_list.append(chunks[i].page_content) 

db = FAISS.from_texts(str_list, embeddings)
db.save_local("cohere_index")