import os
import tempfile
import uuid
from claimreview.keysInstance import keysInstance
from claimreview.vstoreInstance import vstoreInstance
from claimreview.Embeddings import Embeddings
from claimreview.memoryhandler import memoryhandler
from claimreview.chatbot import Chatbot
from fastapi import FastAPI, File, Request, Response, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from typing import AsyncIterable
from pydantic import BaseModel
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


class Message(BaseModel):
    content: str

class TextToSynthesize(BaseModel):
    text: str


keysInstance = keysInstance()
openai_api_key = keysInstance.get_openai_api_key()
pinecone_api_key = keysInstance.get_pinecone_api_key()
pinecone_data_url = keysInstance.get_pinecone_data_url()
index_name = keysInstance.get_index_name()
Embeddings = Embeddings(openai_api_key)
embedder = Embeddings.get_embedding_object()

vstoreInstance = vstoreInstance(pinecone_api_key=pinecone_api_key, index_name=index_name, pinecone_data_url=pinecone_data_url)
idx = vstoreInstance.get_index()
vectorstore = vstoreInstance.get_vector_store(idx, embedder, "text")

claimc = Chatbot(vectorstore=vectorstore, openai_api_key=openai_api_key)
# claimbot = claimc.get_lcel()

app = FastAPI()
openai_client = OpenAI()


origins = [
    "http://localhost:5173","http://localhost:3000",
    "https://electionsgpt.org"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def hello_word():
    return "fcgpt backend is running !!!"

# async def send_message(query: str) -> AsyncIterable[str]:
#     async for s in claimbot.astream({"question": query}):
#         yield s

@app.post("/api/chatbot")
async def chatstream(response: Response, request: Request, question: Message, override_limit: bool = False):
    # request_count = int(request.cookies.get("counter", "0"))
    # if not override_limit and request_count >= 3:
    #     raise HTTPException(status_code=429, detail="Request limit exceeded")
    # request_count += 1
    # streaming_response = StreamingResponse(send_message(question.content), media_type="text/event-stream")
    response = claimc.get_lcel(question.content)
    # streaming_response.headers.append('Set-Cookie', f'counter={request_count}')
    content = response.content if response else ''
    print("content : ", content)
    return {"content": content}

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        print("unique_filename : ", unique_filename)
        file_path = os.path.join("./search_queries", unique_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as audio_file:
            audio_file.write(await file.read())
        with open(file_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        print("transcription.text : ", transcription.text)
        os.remove(file_path)
        return transcription.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech-synthesis")
async def speech_synthesis(text_data: TextToSynthesize):
    input_text = text_data.text
    print("text : ", input_text)
    if not input_text:
        raise HTTPException(status_code=400, detail="API: No text provided for synthesis")

    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=input_text
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            response.stream_to_file(tmp_file.name)
            tmp_file_path = tmp_file.name

        return FileResponse(tmp_file_path, media_type="audio/mpeg", filename="speech.mp3")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API: Failed to synthesize speech: {str(e)}")


@app.post("/api/feedback")
async def forward_feedback_to_mail(form_data: dict):
    smtp_server = "smtpout.secureserver.net"
    port = 465
    sender_email = keysInstance.get_email_api_key()
    password = keysInstance.get_password_api_key()
    msg = MIMEMultipart()
    msg.set_unixfrom("author")
    msg["From"] = sender_email
    msg["To"] = "me.piyushaggarwal@gmail.com"
    msg["Subject"] = "Feedback from Factcheck"
    form_data = form_data["form_data"]
    data_from = form_data["from_email"]
    data_message = form_data["message"]
    data_subject = form_data["subject"]
    data_name = form_data["name"]
    message = f"Feedback From Factcheck\nFrom Email: {data_from},\nname: {data_name},\nsubject: {data_subject},\nmessage: {data_message}"
    msg.attach(MIMEText(message))
    try:
        server = smtplib.SMTP_SSL(smtp_server, port)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email, "me.piyushaggarwal@gmail.com", msg.as_string())
    except Exception as e:
        return str(e)
    finally:
        server.quit()
        return "mail sent", 200
