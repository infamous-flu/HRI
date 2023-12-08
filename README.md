# Human-Robot Interaction (HRI)

This README provides a simple guide to set up and run a LLM-RAG system utilizing Ollama, an open-source platform for running large language models locally.

## Prerequisites

Ensure that you have Ollama installed and running in the background before proceeding with the following steps.

## Step 1: Create Chroma Vectorstore

To initialize the Chroma vectorstore, run the following command in your terminal:

```bash
python vectorstore.py
```

This command generates the vectorstore from the data.

## Step 2: Run Chatbot

Once the vectorstore is set up, launch the chatbot using the following command:

```bash
python chatbot.py
```

This command activates the LLM-RAG system, allowing users to engage with the chatbot. The chatbot leverages the capabilities of Mistral-7B model and the Chroma vectorstore to provide responsive and context-aware interactions.
