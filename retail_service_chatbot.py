import sys
from langchain.llms import Ollama
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

embedding = GPT4AllEmbeddings()
persist_directory = "db"
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="mistral", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


qa_system_prompt = """ [INST] You are a helpful, friendly shopping assistant \
for a sneaker store. Use the following pieces of retrieved context to answer \
the question. If you don't know the answer, just say that you don't know. Use \
three sentences maximum and keep the answer concise. [/INST]

{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

condense_q_system_prompt = """ Given a chat history and the latest user \
question which might reference the chat history, formulate a standalone \
question which can be understood without the chat history. Do NOT answer \
the question, just reformulate it if needed and otherwise return it as is.
"""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

chat_history = []
print("\n"*100)
print("AI: Hello! How can I help you? (Ask a question, or CTRL-D to exit.)")
print("Human:", end=" ", flush=True)
for query in sys.stdin:
    if (len(chat_history) == 5):
        chat_history.pop(0)
    ai_msg = rag_chain.invoke(
        {"question": query, "chat_history": chat_history})
    chat_history.extend(
        [HumanMessage(content=query), AIMessage(content=ai_msg)])
    print(f"AI: {ai_msg.strip()}")
    print("Human:", end=" ", flush=True)
print("\n\n")
print(chat_history)
