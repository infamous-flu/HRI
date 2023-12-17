################################################################################################

"""REQUIRED MODULES"""

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

################################################################################################

"""LLM AND CHAT MODEL CONFIGURATION"""

model = "mistral"

chat = ChatOllama(model=model, temperature=0)
llm = Ollama(model=model, temperature=0)

################################################################################################

"""VECTORSTORE RETRIEVER CONFIGURATION"""

persist_directory = "db"
embedding_function = GPT4AllEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

################################################################################################

"""QA PROMPT CONFIGURATION"""

# TODO: CHANGE THE SYSTEM PROMPT HERE TO SEE IF WE GET BETTER RESPONSESs

qa_system_prompt = """You are an expert AI shopping assistant for our sneaker store, equipped with a powerful retriever system to find \
the most relevant sneakers in our database based on the customer's question. Your primary goal is to skilfully guide customers towards \
the perfect purchase. Utilize compelling persuasion, social proof, and relatable language to encourage customer purchases. Ensure your \
recommendations align with the sneakers available in our store. Politely decline responses if confronted with offensive questions. For \
off-topic inquiries, steer the conversation back to sneakers. Prioritize positive customer experiences and offer creative styling opinions \
grounded within the retrieved context. Keep your responses brief (under 80 words) and always end with a follow-up question to encourage \
continued interaction. 

{context}
"""
qa_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

################################################################################################

"""CONDENSE QUESTION PROMPT CONFIGURATION"""

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

################################################################################################

"""CONDENSE QUESTION CHAIN CONFIGURATION"""

condense_q_chain = condense_q_prompt | chat | StrOutputParser()

################################################################################################

"""HELPER FUNCTIONS"""


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


################################################################################################

"""RAG CHAIN CONFIGURATION"""

rag_chain = (
    RunnablePassthrough.assign(
        context=condense_question | retriever | format_docs)
    | qa_prompt
    | chat
    | StrOutputParser()
)

################################################################################################

"""MESSAGES AND PROMPTS"""

# TODO: WRITE A BETTER WELCOME MESSAGE AND FIRST TIME MESSAGE!

# WELCOME MESSAGE : INFORMATION ABOUT THE ROBOT AND STORE, CREATE ILLUSION OF FRIENDLINESS
# FIRST TIME MESSAGE: INSTRUCTIONS TO HUMAN ON HOW TO INTERACT WITH THE ROBOT AND WHAT TO ASK

first_human_msg = """You are an expert AI shopping assistant. Your primary goal is to expertly persuade me \
in purchasing the perfect sneakers. Use compelling persuasive skills, social proof, and relatable language \
to encourage me to make a purchase. Ensure your recommendations align with the sneakers available in the store. \
If I ask offensive questions, politely decline to answer. If I ask non-sneaker-related questions, steer the \
conversation back to sneakers. Keep your responses concise (under 80 words) and always end with a follow-up \
question to encourage continued interaction. Now greet me and provide instructions on how to interact with you.
"""

welcome_msg = """I am your NAO shopping assistant. Feel free to ask me any questions you have about our fabulous \
collection. I can help you find the perfect pair of sneakers that match your style and preferences."""

instructions_msg = """ You have 10 seconds for each of your questions, please keep your questions short and \
concise. Wait half a second after each of my responses before asking your question to ensure optimal speech \
recognition. My eyes will light up green when I'm listening and red when I've stopped. Say "goodbye" if you \
want to stop the interaction. Do you have a question?
"""

goodbye_msg = """AI: Goodbye, we hope you find the perfect sneakers for your needs at \
our store. If you have any questions or need assistance, feel free to reach out again.
"""

################################################################################################

"""MEMORY CONFIGURATION"""

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=400, return_messages=True
)
memory.save_context({"input": first_human_msg}, {
                    "output": welcome_msg+instructions_msg})

################################################################################################

"""MAIN FUNCTION"""


def main():
    print("\n"*100)
    print("AI: "+welcome_msg+instructions_msg)
    while True:
        chat_history = memory.load_memory_variables({}).get("history", [])
        question = input("Human: ")
        if question == "/bye" or question == "":
            break
        if question == "/memory":
            pprint(f"{chat_history}")
            print()
            continue
        if question == "/clear":
            memory.clear()
            memory.save_context(
                {"input": first_human_msg}, {"output": welcome_msg+instructions_msg})
            print("AI: Memory cleared.\n")
            continue

################################################################################################

        # TODO: ADDITIONAL INSTRUCTION APPENDED AFTER EACH HUMAN MESSAGE

        #   BE CAREFUL!!! THIS IS MUCH STRONGER THAN CHANGING THE SYSTEM
        #   PROMPT AND CAN REALLY LEAD TO DUMB AI RESPONSES

        question += " Keep your response under 80 words."

###############################################################################################

        ai_msg = rag_chain.invoke(
            {"question": question, "chat_history": chat_history})
        print(f"AI: {ai_msg}\n")
        memory.save_context({"input": question}, {"output": ai_msg})
    print(goodbye_msg)


if __name__ == "__main__":
    main()
