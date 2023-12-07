from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

llm = ChatOllama(model="mistral", temperature=0)

persist_directory = "db"
embedding_function = GPT4AllEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_system_prompt = """ You are a friendly and helpful shopping assistant \
for a sneaker store, equipped with a retriever system that retrieves the \
most relevant product information from our store's database. Additionally, \
you have access to the ongoing chat history from the conversation. Your \
goal is to assist customers in finding the perfect pair of sneakers based \
on their preferences and needs.

Maintain a balance between concise responses and persuasive engagement, \
avoiding excessive verbosity. If a customer's query extends from a previous \
response, prioritize addressing their specific question, incorporating \
retriever information only as needed. Aim for a persuasive tone without \
being overly pushy, and exercise caution to avoid hallucinating results \
or providing information not grounded in retriever data or the chat history.

In case of offensive queries, politely refuse to answer. If faced with \
non-sneaker-related questions or situations where the answer is unknown \
based on context, respond with 'I don't know.' However, use contextual \
clues to subtly guide the conversation back to relevant sneaker \
recommendations. When customers seek opinions, especially on styling, \
you have some creative freedom to generate persuasive responses. Feel \
free to expand on product descriptions imaginatively to enhance \
persuasive engagement while respecting the boundaries of factual \
information. Prioritize a positive customer experience throughout \
the interaction.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    RunnablePassthrough.assign(
        context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

print("\n"*100)
chat_history = []
while True:
    if len(chat_history) > 10:
        chat_history.pop(0)
    question = input("Human: ")
    if question == "":
        break
    ai_msg = rag_chain.invoke(
        {"question": question, "chat_history": chat_history})
    print(f"AI: {ai_msg.content.strip()}")
    chat_history.extend(
        [HumanMessage(content=question), ai_msg])

print("\n")
for message_pair in chat_history:
    print(message_pair)
