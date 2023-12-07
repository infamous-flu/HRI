import json
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

with open("data/sneakers.json") as file:
    sneakers = json.load(file)

documents = []
for sneaker in sneakers:
    document = Document(
        page_content=sneaker["description"],
        metadata={
            "name": sneaker["name"],
            "brand": sneaker["brand"],
            "price": sneaker["price"]
        }
    )
    documents.append(document)

embedding = GPT4AllEmbeddings()
persist_directory = "db"
Chroma.from_documents(documents=documents, embedding=embedding,
                      persist_directory=persist_directory)
