from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
import random
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_directory = '/home/bharath/project/Transformer'
text_files = [os.path.join(text_directory, file) for file in os.listdir(text_directory) if file.endswith('.txt')]

for text_file in text_files:
    print(f'this is {text_file}')
    # Load text data from a file using TextLoader
    loader = TextLoader(text_file)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)


qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    prefer_grpc=True,
    collection_name="myfiles",
    # location=":memory:", 
     path="./qdrant_db",
    force_recreate=True,#Currently, the collection is going to be reused if it already exists. Setting force_recreate to True allows to remove the old collection and start from scratch.
)
query = "who is ganesh"
result = qdrant.similarity_search(query)
print(result)