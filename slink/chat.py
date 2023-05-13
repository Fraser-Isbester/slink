"""Chat auto complete."""

import json
import logging
import os

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

logging.basicConfig(level=logging.DEBUG)


BAD_RESPONSES = [
    "I'm not able to help with that, as I'm only a language model. If you believe this is an error, please send us your feedback."
]


def main():

    chat_file = "chats/chat.ndjson"

    user = "fraser"
    context = f"""
        You are {user} a helpful engineering manager of Data Infrastructure at
        Virta Health who responds to slack messages in a single sentence.
    """
    messages = load_messages(chat_file)

    logging.debug("Loaded %d messages from %s", len(messages), chat_file)

    # If no question, exit 0
    if messages[-1]["user"] == user:
        return

    logging.debug("Last message is from '%s' generating response.", user)

    msg_strs = [json.dumps(msg) for msg in messages]

    logging.debug("Processing %s", msg_strs[-1])
    response = gen_response(context, msg_strs[:-1], msg_strs[-1])


    if response in BAD_RESPONSES:
        raise ValueError("Bad response from model: %s", response)

    add_message(chat_file, response)


def add_message(chat_file, msg):
    """adds a new message to the chat."""


    os.makedirs(os.path.dirname(chat_file), exist_ok=True)
    with open(chat_file, "a", encoding="utf-8") as chat:
        msg_structured = {
            "user": "fraser",
            "user_type": "ai",
            "text": msg
        }

        msg_string = json.dumps(msg_structured)
        chat.writelines([msg_string+"\n"])

def load_messages(chat_file) -> list:
    """loads all messages from the chat."""

    os.makedirs(os.path.dirname(chat_file), exist_ok=True)
    with open(chat_file, "r", encoding="utf-8") as chat:
        lines = chat.readlines()

        return [json.loads(line.strip()) for line in lines]


def gen_response(
    context: str,
    examples: list,
    latest: str,
    project_id: str ,
    model_name: str = "chat-bison@001",
    temperature: float = 0.2,
    max_output_tokens: int = 256,
    top_p: float = 0.8,
    top_k: int = 40,
    location: str = "us-central1",
) -> str:
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)

    chat_model = ChatModel.from_pretrained(model_name)
    parameters = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
        "top_k": top_k,
    }

    example_msgs = [json.loads(msg)["text"] for msg in examples]
    pairs = [chunk for chunk in chunks(example_msgs, 2)]
    chat_pairs = [InputOutputTextPair(pair[0], pair[1]) for pair in pairs]

    chat = chat_model.start_chat(
        context=context,
        examples=chat_pairs,
    )
    response = chat.send_message(latest, **parameters)

    return response.text


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    main()
