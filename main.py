import wave
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

welcome_msg_short = """Welcome to our sneaker store! I'm your NAO shopping \
assistant. I can help you find the perfect sneaker, answer your questions, \
and provide personalized recommendations. Ask me anything about sneakers!
"""

response_time = 6
first_time_msg = f""" You have {response_time} seconds for each of your \
    questions. So please keep your questions short and succinct. Please \
    also wait half a second after each of my responses before you start \
    asking your questions to ensure optimal speech recognition. My eyes \
    will light up green when I'm listening and red when I've stopped. You \
    can also say 'goodbye' to stop this interaction."""

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=300, return_messages=True
)
memory.save_context({"input": ""}, {"output": welcome_msg_long})


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
    yield session.call("rie.dialogue.say", text=welcome_msg_short+first_time_msg)
    answer = ""
    while True:
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 255, 0]}}],)
        print("Listening...")
        frames = yield session.call("rom.sensor.hearing.read", time=response_time*1000)
        audio_data = b""
        for frame in frames:
            audio_data += frame["data"].get("body.head.front", b"")
        with open("output.raw", "wb") as raw_file:
            raw_file.write(audio_data)
        print("Stopped listening")
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [255, 0, 0]}}],)
        question = transcribe_file(f"output.raw")
        if "goodbye" in question:
            break
        if question == "":
            yield session.call("rom.actuator.audio.stop")
            answer = "Sorry, I didn't quite catch what you were saying. Can you speak more clearly?"
            session.call("rom.optional.behavior.play", name="BlocklyShrug")
            yield session.call("rom.actuator.light.write", mode="linear", frames=[
                {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
            yield session.call("rie.dialogue.say", text=answer)
            continue
        question += " Keep your response under 80 words."
        chat_history = memory.load_memory_variables({}).get("history", [])
        ai_msg = rag_chain.invoke(
            {"question": question, "chat_history": chat_history})
        answer = ai_msg.content
        print(answer)
        memory.save_context({"input": question}, {"output": answer})
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
        yield session.call("rie.dialogue.say", text=answer)
    yield session.call("rom.actuator.light.write", mode="linear", frames=[{"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
    goodbye_msg = """Goodbye! I hope you find the perfect sneakers for your needs \
    at our store. If you have any more questions or need further assistance, feel \
    free to reach out again.
    """
    yield session.call("rie.dialogue.say", text=goodbye_msg)
    session.call("rom.optional.behavior.play", name="BlocklyBow")
    session.leave()
    pprint(chat_history)


wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
                "serializers": ["msgpack"],
                "max_retries": 0
                }],
    realm="rie.657ac934cfc130d68e543e5b",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])
