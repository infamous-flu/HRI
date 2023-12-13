import csv
from pprint import pprint
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Assuming your CSV file has columns: "name", "brand", "price", and "description"
csv_file_path = "data/sneakers.csv"

documents = []
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        document = Document(
            page_content=row["Description"],
            metadata={
                "name": row["Name"],
                "brand": row["Brand"],
                "price": int(row["Price"])
            }
        )
        documents.append(document)

embedding = GPT4AllEmbeddings()
persist_directory = "db"
Chroma.from_documents(documents=documents, embedding=embedding,
                      persist_directory=persist_directory)
