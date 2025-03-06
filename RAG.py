from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import GroqEmbeddings
import getpass
import os

# Directory containing your PDF files
DATA_PATH = "/Users/bisman/Documents/ResInsights/Raw Data/"

def load_documents():
    # Ensure the path is a directory and not a file pattern
    if os.path.isdir(DATA_PATH):
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()
    else:
        raise ValueError(f"The path {DATA_PATH} is not a valid directory.")

# print(documents[0])  # Print the first loaded document

def split_documents(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex= False
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)
print(chunks[0])

