import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Define the base model with a Stop Sequence
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=50,  # Reduced since we only want a short answer
    temperature=0.1,
    # This stops the model from rambling after it gives the answer
    stop_sequences=["</s>", "<|user|>", "\n\n"],
)

# 2. Wrap it in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

try:
    print("Sending request to Zephyr-7B...")
    # Be very direct with Zephyr
    messages = [
        HumanMessage(content="Response: 'Hello'. Task: Say hello in exactly one word.")
    ]

    response = chat_model.invoke(messages)

    print("-" * 30)
    # Strip whitespace to keep it clean
    print(f"Hugging Face Response: {response.content.strip()}")
    print("-" * 30)

except Exception as e:
    print(f"\n[Error]: {e}")
