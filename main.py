from claimreview.keysInstance import keysInstance
from claimreview.vstoreInstance import vstoreInstance
from claimreview.Embeddings import Embeddings
from claimreview.memoryhandler import memoryhandler
from claimreview.chatbot import Chatbot
from fastapi import FastAPI, Request, Response, HTTPException
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from fastapi.responses import StreamingResponse
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from typing import AsyncIterable
from pydantic import BaseModel
import asyncio
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class Message(BaseModel):  # pydantic object for message input
    content: str


# Initialize keys
keysInstance = keysInstance()
openai_api_key = keysInstance.get_openai_api_key()
pinecone_api_key = keysInstance.get_pinecone_api_key()


# Initialize embeddings
Embeddings = Embeddings(openai_api_key)
embedder = Embeddings.get_embedding_object()

# Initialize index and vectorstore
vstoreInstance = vstoreInstance(pinecone_api_key)
idx = vstoreInstance.get_index()
vector_store = vstoreInstance.get_vector_store(idx, embedder, "text")

# Initialize MemoryHandler
# memoryhandler = memoryhandler()
# memory = memoryhandler.get_memory()

# import prompts

# prompt = Prompt()
# condensed_question_template = prompt.get_prompt()

# qa_chain_initializer = chainhandler(openai_api_key, embedder, memory)

claimc = Chatbot(vector_store=vector_store, openai_api_key=openai_api_key)

claimbot = claimc.get_lcel()

app = FastAPI()


@app.get("/api/status")  # return api status
async def hello_word():
    return "fcgpt backend is running !!!"


# async def send_message(query: str) -> AsyncIterable[str]:      #returns streamed output from chatbot
#     callbackhandler = AsyncIteratorCallbackHandler()
#     model = qa_chain_initializer.get_model(callback_handler=callbackhandler)
#     qachain = ConversationalRetrievalChain.from_llm(
#         llm=model,
#         retriever=vector_store.as_retriever(),
#         condense_question_prompt=condensed_question_template,
#         memory=memory,
#         return_source_documents=True
#     )

#     task = asyncio.create_task(
#         qachain.acall({
#             "question" : query
#         }, return_only_outputs=True)
#     )
#     try:
#         async for token in callbackhandler.aiter():
#             yield token
#     except Exception as e:
#         print(f"Exception caught : {e}")
#     finally:
#         callbackhandler.done.set()

#     await task
#     callbackhandler.done.set()

# @app.post("/api/chatbot/")
# async def chatstream(question: Message):
#     answer = send_message(question.content)
#     return StreamingResponse(
#         answer,media_type="text/event-stream"
#     )


async def send_message(query: str) -> AsyncIterable[str]:
    async for s in claimbot.astream({"question": query}):
        yield s


@app.post("/api/chatbot")
async def chatstream(response:Response, request:Request,question: Message,override_limit: bool = False):
    request_count = int(request.cookies.get("counter","0"))
    if not override_limit and request_count >= 3:
        raise HTTPException(status_code=429, detail="Request limit exceeded")
    request_count += 1
    streaming_response = StreamingResponse(send_message(question.content), media_type="text/event-stream")
    
    # Set the cookie directly on the streaming response
    streaming_response.headers.append('Set-Cookie', f'counter={request_count}')
    return streaming_response


@app.post("/api/feedback")
async def forward_feedback_to_mail(form_data: dict):
    smtp_server = "smtpout.secureserver.net"
    port = 465  # For starttls
    sender_email = keysInstance.get_email_api_key()
    password = keysInstance.get_password_api_key()
    # Create a secure SSL context
    msg = MIMEMultipart()
    msg.set_unixfrom("author")
    msg["From"] = sender_email
    msg["To"] = "me.piyushaggarwal@gmail.com"
    msg["Subject"] = "Feedback from Factcheck"
    form_data = form_data["form_data"]
    data_from = form_data["from_email"]
    data_messege = form_data["messege"]
    data_subject = form_data["subject"]
    data_name = form_data["name"]

    message = f"Feedback From Factcheck\n From Email:{data_from},\n name:{data_name},\n subject:{data_subject},\n messege:{data_messege}"
    msg.attach(MIMEText(message))
    # Try to log in to server and send email
    try:
        server = smtplib.SMTP_SSL(smtp_server, port)
        server.ehlo()  # Can be omitted
        # server.starttls(context=context)  # Secure the connection
        # server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, "me.piyushaggarwal@gmail.com", msg.as_string())
        # TODO: Send email here
    except Exception as e:
        # Print any error messages to stdout
        return e
    finally:
        server.quit()
        return "mail sent", 200
