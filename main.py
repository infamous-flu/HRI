################################################################################################

"""REQUIRED MODULES"""

import os
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
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

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

goodbye_msg = """Goodbye, we hope you find the perfect sneakers for your needs at our store. If you have any \
questions or need assistance, feel free to reach out again."""

################################################################################################

"""MEMORY CONFIGURATION"""

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=300, return_messages=True
)
memory.save_context(
    {"input": first_human_msg}, {"output": welcome_msg+instructions_msg})

################################################################################################

"""SPEECH RECOGNITION FUNCTION"""


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

################################################################################################


"""MAIN FUNCTION"""


@inlineCallbacks
def main(session, details):

    ############ INITIALIZATION ############

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

    ############# INTRODUCTION #############

    # Play the BlocklyStand behavior to set the initial pose
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    # Set the eyes' color to blue
    yield session.call("rom.actuator.light.write", mode="linear", frames=[
        {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
    # Configure the dialogue system to use the English (UK) language
    yield session.call("rie.dialogue.config.language", "en_uk")
    # Set the audio volume to maximum (100)
    yield session.call("rom.actuator.audio.volume", 100)
    # Play a behavior (BlocklyWaveRightArm) to wave at the customer
    session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
    # Say a welcome message to the customer
    yield session.call("rie.dialogue.say", text="Welcome to our sneaker store!")
    yield session.call("rie.dialogue.say_animated", text=welcome_msg+instructions_msg)

    ############## MAIN LOOP ###############

    while True:
        # Set the eyes' color to green while listening
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 255, 0]}}],)
        # Print a message indicating that the system is listening
        print("Listening...")
        # Read audio frames from the sensor for a specified time duration (8 seconds)
        frames = yield session.call("rom.sensor.hearing.read", time=8000)
        # Combine audio data from all frames into a single byte string
        audio_data = b""
        for frame in frames:
            audio_data += frame["data"].get("body.head.front", b"")
        # Write the combined audio data to a raw audio file ("output.raw")
        with open("output.raw", "wb") as raw_file:
            raw_file.write(audio_data)
        # Print a message indicating that the system has stopped listening
        print("Stopped listening")
        # Set the eyes' color to red after listening
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [255, 0, 0]}}],)
        # Transcribe the recorded audio file to obtain the user's question
        question = transcribe_file(f"output.raw")
        # Check if the user wants to end the conversation
        if "goodbye" in question or "bye" in question:
            break
        # Check if the user wants to clear the memory (for testing only)
        if "clear memory" in question:
            memory.clear()
            answer = "Memory cleared"
            memory.save_context(
                {"input": first_human_msg}, {"output": welcome_msg+instructions_msg})
            # Set the eyes' color to blue
            yield session.call("rom.actuator.light.write", mode="linear", frames=[
                {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
            yield session.call("rie.dialogue.say", text=answer)
            print(answer)
            continue
        # If no question is detected, prompt the user to speak louder and more clearly
        if question == "":
            # Prompt the user to speak louder and more clearly
            answer = "Sorry, I didn't quite catch what you were saying. Can you speak louder and more clearly?"
            # Play a behavior (BlocklyShrug) to express confusion or uncertainty
            session.call("rom.optional.behavior.play", name="BlocklyShrug")
            # Set the eyes' color to blue
            yield session.call("rom.actuator.light.write", mode="linear", frames=[
                {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
            # Speak the apology to the user
            yield session.call("rie.dialogue.say", text=answer)
            print(answer)
            # Continue to the next iteration of the loop
            continue

        ########################################################################################

        """ADDITIONAL INSTRUCTION APPENDED AFTER EACH HUMAN MESSAGE"""

        # TODO: MAYBE CHANGE THIS TO GET A BETTER RESPONSE FROM THE AI?

        #   BE CAREFUL!!! THIS IS MUCH STRONGER THAN CHANGING THE SYSTEM
        #   PROMPT AND CAN REALLY LEAD TO DUMB AI RESPONSES

        question += " Keep your response under 80 words."

        ########################################################################################

        # Load chat history from memory variables
        chat_history = memory.load_memory_variables({}).get("history", [])
        # Use the chat model (rag_chain) to generate a response based on the user's question
        answer = rag_chain.invoke(
            {"question": question, "chat_history": chat_history})
        # Print the generated answer to the console
        print(answer)
        # Save the user's question and the generated answer to the context memory
        memory.save_context({"input": question}, {"output": answer})
        # Set the eyes' color to blue
        yield session.call("rom.actuator.light.write", mode="linear", frames=[
            {"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
        # Speak the generated answer in an animated manner
        yield session.call("rie.dialogue.say_animated", text=answer)

    ############### GOODBYE ################

    # Set the eyes' color to blue
    yield session.call("rom.actuator.light.write", mode="linear", frames=[{"time": 0000, "data": {"body.head.eyes": [0, 0, 255]}}],)
    # Speak the goodbye message in an animated manner
    yield session.call("rie.dialogue.say_animated", text=goodbye_msg)
    # Play the BlocklyBow behavior to perform a bowing motion
    session.call("rom.optional.behavior.play", name="BlocklyBow")
    # End the session
    session.leave()
    # Print the chat history to the console
    pprint(chat_history)


################################################################################################

"""WAMP CONFIGURATION"""

wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
                "serializers": ["msgpack"],
                "max_retries": 0
                }],
    realm=os.getenv("WAMP_REALM"),
)

wamp.on_join(main)

################################################################################################


if __name__ == "__main__":
    run([wamp])
