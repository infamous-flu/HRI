import json
from autobahn.twisted.component import Component, run
from autobahn.wamp.exception import ApplicationError
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
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
from random import choice
from pprint import pprint

model = "mistral"

chat = ChatOllama(model=model, temperature=0.1)
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

#################################################################################

# TODO: WRITE A BETTER WELCOME MESSAGE AND FIRST TIME MESSAGE!

#################################################################################

welcome_msg = """I am your NAO shopping assistant. Feel free to ask me \
any questions you have about our fabulous collection. I can help you find the \
perfect pair of sneakers that match your style and preferences.
"""

first_time_msg = f""" You have 10 seconds for each of your questions, please \
keep your questions short and concise. Wait half a second after each of my \
responses before asking your question to ensure optimal speech recognition. \
My eyes will light up green when I'm listening and red when I've stopped. \
Say "goodbye" if you want to stop the interaction. Do you have a question?"""

#################################################################################

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=400, return_messages=True
)
memory.save_context(
    {"input": ""}, {"output": welcome_msg+first_time_msg})


def transcribe_file(speech_file: str):
    try:
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
        print(output.strip())
        return output.strip()
    except Exception as e:
        return ""


@inlineCallbacks
def main(session, details):
    print("Starting up...")
    print("Waiting for cloud modules")
    ready = False
    while not ready:
        try:
            yield session.call("rie.cloud_modules.ready")
            ready = True
        except ApplicationError:
            print("Cloud modules are not ready yet")
            yield sleep(0.25)
    print("Cloud modules are ready to go!")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    yield session.call("rom.actuator.light.write", mode="linear", frames=[
        {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
    yield session.call("rie.dialogue.config.language", "en_uk")
    yield session.call("rom.actuator.audio.volume", 100)
    session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
    yield session.call("rie.dialogue.say", text="Welcome to our sneaker store!")
    yield session.call("rie.dialogue.say_animated", text=welcome_msg+first_time_msg)
    answer = ""
    while True:
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 255, 0]}}],)
        print("Listening...")
        frames = yield session.call("rom.sensor.hearing.read", time=8000)
        audio_data = b""
        for frame in frames:
            audio_data += frame["data"].get("body.head.front", b"")
        with open("output.raw", "wb") as raw_file:
            raw_file.write(audio_data)
        print("Stopped listening")
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [255, 0, 0]}}],)
        question = transcribe_file(f"output.raw")
        if "goodbye" in question or "bye" in question:
            break
        if question == "":
            yield session.call("rom.actuator.audio.stop")
            answer = "Sorry, I didn't quite catch what you were saying. Can you speak louder and more clearly?"
            session.call("rom.optional.behavior.play", name="BlocklyShrug")
            yield session.call("rom.actuator.light.write", mode="linear", frames=[
                {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
            yield session.call("rie.dialogue.say", text=answer)
            continue
        question += " Keep your response under 80 words."
        chat_history = memory.load_memory_variables({}).get("history", [])
        answer = rag_chain.invoke(
            {"question": question, "chat_history": chat_history})
        print(answer)
        memory.save_context({"input": question}, {"output": answer})
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
        session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
        yield session.call("rie.dialogue.say_animated", text=answer)
    yield session.call("rom.actuator.light.write", mode="linear", frames=[{"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
    goodbye_msg = """Goodbye! I hope you find the perfect sneakers for your needs \
    at our store. If you have any more questions or need further assistance, feel \
    free to reach out again.
    """
    yield session.call("rie.dialogue.say_animated", text=goodbye_msg)
    session.call("rom.optional.behavior.play", name="BlocklyBow")
    session.leave()
    pprint(chat_history)


wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
                "serializers": ["msgpack"],
                "max_retries": 0
                }],
    realm="rie.657c29b3cfc130d68e544a00",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])
