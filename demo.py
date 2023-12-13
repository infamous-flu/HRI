from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from google.cloud import speech
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

model = "neural-chat"

chat = ChatOllama(model=model, temperature=0)
llm = Ollama(model=model, temperature=0)

persist_directory = "db"
embedding_function = GPT4AllEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_system_prompt = """<<SYS>> You are a friendly AI shopping assistant for \
our sneaker store, equipped with a powerful retriever system and unmatched \
persuasion skills. Your primary goal is to assist customers in finding the \
perfect pair of sneakers from our collection based on customer preferences.\
Simultaneously, seize every opportunity to leverage your expert persuasive \
abilities to encourage them to make a purchase, maintaining a friendly and \
non-intrusive approach.

Ensure that all recommendations align with the sneakers available in our \
store, avoiding any fictional or out-of-context suggestions. Make use of \
the ongoing chat history for context and verification purposes. If faced \
with offensive questions, politely decline to answer. If the question is \
not related to sneakers, guide the conversation back to relevant sneaker \
recommendations.

Prioritize creating a positive customer experience throughout the interaction. \
You have creative freedom to provide opinions on styling, but stay grounded in \
the retrieved context and chat history. Your focus is on helping customers not \
only discover the right sneakers, but also ensuring that they are persuaded to \
purchase our product. <</SYS>>

<<CONTEXT>> {context} <</CONTEXT>>
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
)

welcome_msg_long = """
Hey Sneaker Enthusiast! Welcome to our ultimate sneaker haven! I am your \
trusty shopping assistant, ready to assist you on a journey to find the \
perfect pair of sneakers that match your style and preferences. Whether \
you're after comfort, crushing on the latest trends, or searching for a \
classic look, I have got your back. Let's lace up and explore the world \
of sneakers together! Feel free to ask me anything about our fabulous \
collection, and let's kick off this sneaker adventure with a burst of \
excitement!
"""

welcome_msg_short = """Welcome to our sneaker store, how may I help you?"""

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=300, return_messages=True
)
memory.save_context({"input": ""}, {"output": welcome_msg_short})


def transcribe_file(speech_file: str):
    client = speech.SpeechClient()
    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    output = ""
    for result in response.results:
        output += result.alternatives[0].transcript
    print(output)
    return output


@inlineCallbacks
def main(session, details):
    response_time = 8
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    yield session.call("rie.dialogue.config.language", "en_uk")
    yield session.call("rom.actuator.audio.volume", 90)
    first_time_msg = f""" You have {response_time} seconds for each of your \
        responses. Please also wait half a second after each of my response \
        to ensure optimal speech recognition."""
    answer = welcome_msg_short + first_time_msg
    session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
    yield session.call("rie.dialogue.say", text=answer)
    while True:
        session.call("rom.optional.behavior.play", name="BlocklyTouchHead")
        print("Listening...")
        frames = yield session.call("rom.sensor.hearing.read", time=response_time*1000)
        audio_data = b""
        for frame in frames:
            audio_data += frame["data"].get("body.head.front", b"")
        with open("output.raw", "wb") as raw_file:
            raw_file.write(audio_data)
        print("Stopped listening")
        question = transcribe_file(f"output.raw")
        if "stop" in question:
            break
        question += " INSTRUCTION: YOUR RESPONSE MUST BE CONCISE! DO NOT EXCEED 80 WORDS!"
        chat_history = memory.load_memory_variables({}).get("history", [])
        ai_msg = rag_chain.invoke(
            {"question": question, "chat_history": chat_history})
        answer = ai_msg.content
        print(answer)
        memory.save_context({"input": question}, {"output": answer})
        yield session.call("rie.dialogue.say", text=answer)
    goodbye_msg = """Goodbye! I hope you find the perfect sneakers for your needs \
    at our store. If you have any more questions or need further assistance, feel \
    free to reach out again.
    """
    yield session.call("rie.dialogue.say", text=goodbye_msg)
    session.leave()


wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
                "serializers": ["msgpack"],
                "max_retries": 0
                }],
    realm="rie.65799168cfc130d68e5435ce",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])
