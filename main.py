import time
import sounddevice as sd

import record

from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from kokoro import KPipeline
import threading
import queue

tts_queue = queue.Queue()


# -----------------------------
# LLM
# -----------------------------
llm = ChatOllama(
    model="minimax-m3:cloud",
    streaming=True,
)

checkpointer  = InMemorySaver()

# -----------------------------
# Kokoro TTS
# -----------------------------
tts = KPipeline(lang_code="a")



def tts_worker():
    while True:
        text = tts_queue.get()

        if text is None:
            break

        record.stop_recording()

        try:
            generator = tts(text, voice="af_heart")

            for _, _, audio in generator:
                sd.play(audio, samplerate=24000)
                sd.wait()

        finally:
            record.start_recording()

        tts_queue.task_done()



threading.Thread(
    target=tts_worker,
    daemon=True
).start()


def speak(text):
    """Stop microphone, speak, then restart microphone."""

    if not text.strip():
        return

    # Stop microphone recording
    record.stop_recording()

    try:
        generator = tts(text, voice="af_heart")

        for _, _, audio in generator:
            sd.play(audio, samplerate=24000)
            sd.wait()

        sd.stop()

    finally:
        # Small delay so speaker finishes completely
        time.sleep(0.3)

        # Resume microphone
        record.start_recording()


# -----------------------------
# LangGraph
# -----------------------------
class ModelState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ModelState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ModelState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer= checkpointer)


config = {
    'configurable' :
    {
        'thread_id' : "11"
    }
}


workflow.update_state(config, {'messages' : [SystemMessage(content="Just give answer in one sentence")]})

# -----------------------------
# Chat Loop
# -----------------------------
while True:

    user_text = record.give_text()

    if not user_text.strip():
        continue

    print(f"\nYou: {user_text}")
    print("Assistant: ", end="", flush=True)

    buffer = ""

    for chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content=user_text)]},
        stream_mode="messages",
        config=config,
    ):

        if not chunk.content:
            continue

        token = chunk.content

        print(token, end="", flush=True)

        buffer += token

        # Speak completed sentence immediately
        if any(buffer.endswith(x) for x in [".", "!", "?", "\n"]):
            speak(buffer)
            buffer = ""

        # Or if it gets long, don't wait forever
        elif len(buffer) >= 80:

            idx = buffer.rfind(" ")

            if idx == -1:
                idx = len(buffer)

            speak(buffer[:idx])

            buffer = buffer[idx:].lstrip()

    # Speak remaining text
    if buffer.strip():
        speak(buffer)

    # Wait until queued speech finishes
    tts_queue.join()

    print()