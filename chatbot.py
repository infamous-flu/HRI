from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate
)
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory
from pprint import pprint

model = "mistral"

chat = ChatOllama(model=model, temperature=0)
llm = Ollama(model=model, temperature=0)

persist_directory = "db"
embedding_function = GPT4AllEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_system_prompt = """As an expert AI shopping assistant specializing in sneakers, leverage \
your persuasive skills and our powerful retriever system to enhance customer engagement and \
boost purchases. Employ social proof, scarcity, and relatable language to create a sense of \
urgency and exclusivity. Ensure all recommendations align with our store's inventory. Politely \
handle offensive or non-sneaker-related queries, redirecting the conversations to sneakers. \
Prioritize positive customer experiences and offer creative styling opinions within the context \
of our inventory. Keep responses concise (under 80 words) and always conclude with a follow-up \
question to encourage continued interaction.

{context}
"""
qa_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

condense_q_system_prompt = """Given a chat history and the latest user question \
which may reference the chat history, formulate a standalone question which can \
be understood without the chat history. Do NOT answer the question, just reformulate \
it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
condense_q_chain = condense_q_prompt | chat | StrOutputParser()


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
    | chat
    | StrOutputParser()
)

welcome_msg_long = """
AI: Hey Sneaker Enthusiast! Welcome to our ultimate sneaker haven! I am \
your trusty shopping assistant, ready to guide you on a journey to find \
the perfect pair of sneakers that match your style and preferences. Whether \
you're after comfort, crushing on the latest trends, or searching for a \
classic look, I have got your back. Let's lace up and explore the world \
of sneakers together! Feel free to ask me anything about our fabulous \
collection, and let's kick off this sneaker adventure with a burst of \
excitement!
"""

welcome_msg_short = "Welcome to our sneaker store, how may I help you?"

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=500, return_messages=True
)
memory.save_context({"input": ""}, {"output": welcome_msg_short})

print("\n"*100)
print(welcome_msg_long)
while True:
    chat_history = memory.load_memory_variables({}).get("history", [])
    question = input("Human: ")
    if question == "":
        break
    if question == "/memory":
        pprint(f"{chat_history}\n")
        continue
    question += " Keep your response under 80 words."
    ai_msg = rag_chain.invoke(
        {"question": question, "chat_history": chat_history})
    answer = ai_msg.content.strip().replace("\n\n", "\n").replace("\n", " ")
    print(f"AI: {answer}\n")
    memory.save_context({"input": question}, {"output": answer})

goodbye_msg = """
AI: Goodbye, we hope you find the perfect sneakers for your needs at our store. \
If you have any questions or need assistance, feel free to reach out again.
"""
print(goodbye_msg)
