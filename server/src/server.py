import jwt
import os
import pg8000
import asyncio
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from langcodes import Language

import parse_rss
import llm
import tts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fetch environment variables
MAX_STORIES = 6
DB_USERNAME = os.environ.get("DB_USERNAME")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
if DB_USERNAME is None or DB_PASSWORD is None:
    raise Exception("DB_USERNAME or DB_PASSWORD env vars not set!")

if len(os.environ.get("JWT_SECRET_KEY", "")) == 0:
    print("jwt secret key not found")
    exit()

RESPONSE_MESSAGES = {
    "invalid_auth": "Invalid authentication token!",
    "valid_auth": "Token is valid. Welcome back!",
}


class User(BaseModel):
    email: str
    password: str
    lang: str = "english"


class SourceJson(BaseModel):
    source: str


def get_db():
    host, port = os.environ.get("DB_SERVER").split(":")
    return pg8000.connect(
        host=host,
        port=port,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database="news_briefer",
    )


def check_auth(token: str):
    if not token:
        return False
    try:
        decoded = jwt.decode(
            token, os.environ.get("JWT_SECRET_KEY"), algorithms=["HS256"]
        )
        if decoded.get("email") is not None:
            return decoded
        return False
    except jwt.InvalidTokenError:
        return False


def get_user_sources(email: str):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select * from sources where email = %s", (email,))
    results = cursor.fetchall()
    return [source for _, source in results]


async def get_all_sources_summary_phrases(email: str, lang: str):
    sources = get_user_sources(email)
    items_per_src = MAX_STORIES // len(sources)

    # fetch stories from all sources
    # gives a list of lists, such that dimensions are sources and stories respectively
    news_stories = await asyncio.gather(
        *[parse_rss.get_topn_articles(source, items_per_src + 1) for source in sources]
    )

    # merge all stories into one list
    news_stories_old = news_stories
    news_stories = []
    for i in range(len(news_stories_old)):
        source_stories = news_stories_old[i]
        source_link = sources[i]
        for story in source_stories:
            news_stories.append("".join(["(from ", source_link, ")", story]))

    for story in news_stories:
        async for phrase in llm.summarize_news(story, lang):
            yield phrase


async def get_all_sources_summary_audios(email: str, lang: str):
    first_phrase = True

    async for phrase in get_all_sources_summary_phrases(email, lang):
        if not phrase:
            continue

        # Get binary audio data for the current phrase
        audio = await tts.text_to_audio(phrase, lang)
        audio_io = io.BytesIO(audio)

        # If it's the first phrase, yield the full WAV (header + data)
        if first_phrase:
            # overwrite length bytes in wav header
            # see https://stackoverflow.com/questions/2551943/how-to-stream-a-wav-file
            audio_io.seek(4)
            audio_io.write(b"\xff\xff\xff\xff")
            audio_io.seek(io.SEEK_SET)
            audio_io.seek(40)
            audio_io.write(b"\xff\xff\xff\xff")
            audio_io.seek(io.SEEK_SET)

            yield audio_io.read()  # Yield full WAV with header
            first_phrase = False
        else:
            # For subsequent phrases, skip the 44-byte header
            audio_io.seek(44)
            yield audio_io.read()  # Yield only audio data frames


@app.post("/login")
async def login(user: User):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "select * from users where email = %s and password = %s",
        (user.email, user.password),
    )
    results = cursor.fetchone()
    if results:
        email, password, lang = results

        # convert lang to two-letter code
        lang = Language.find(lang).to_tag()

        token = jwt.encode(
            {"email": email, "lang": lang}, os.environ.get("JWT_SECRET_KEY")
        )
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/signup")
async def signup(user: User):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select * from users where email = %s", (user.email,))
    existing_user = cursor.fetchone()
    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Account with that email already exists! Please log in.",
        )
    cursor.execute(
        "insert into users (email, password, lang) values (%s, %s, %s)",
        (user.email, user.password, user.lang),
    )
    db.commit()
    return {"message": "User created successfully. Log in with your credentials"}


@app.get("/get-headers/{token}")
async def get_headers(token: str):
    decoded = check_auth(token)
    if not decoded:
        raise HTTPException(status_code=401, detail=RESPONSE_MESSAGES["invalid_auth"])
    email = decoded["email"]
    sources = get_user_sources(email)
    items_per_src = MAX_STORIES // len(sources)
    news_headlines = []
    for source in sources:
        news_headlines.extend(parse_rss.get_topn_headlines(source, items_per_src + 1))
    return {"headlines": news_headlines}


@app.get("/get-audio/{token}")
async def get_audio(token: str):
    decoded = check_auth(token)
    if not decoded:
        raise HTTPException(status_code=401, detail=RESPONSE_MESSAGES["invalid_auth"])
    email = decoded["email"]
    lang = decoded["lang"]

    if len(get_user_sources(email)) == 0:
        raise HTTPException(
            status_code=400,
            detail="No sources found. Please add sources before requesting audio.",
        )

    return StreamingResponse(
        get_all_sources_summary_audios(email, lang), media_type="audio/wav"
    )


@app.get("/get-sources/{token}")
async def get_sources(token: str):
    decoded = check_auth(token)
    if not decoded:
        raise HTTPException(status_code=401, detail=RESPONSE_MESSAGES["invalid_auth"])
    email = decoded["email"]
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select * from sources where email = %s", (email,))
    results = cursor.fetchall()
    results = [src for _, src in results]
    return {"sources": results}


@app.post("/add-source/{token}")
async def add_source(token: str, sourcejson: SourceJson):
    decoded = check_auth(token)
    if not decoded:
        raise HTTPException(status_code=401, detail=RESPONSE_MESSAGES["invalid_auth"])
    email = decoded["email"]

    source = sourcejson.source

    db = get_db()
    cursor = db.cursor()
    cursor.execute("insert into sources values (%s, %s)", (email, source))
    db.commit()

    return {"message": "Source added successfully."}


@app.post("/remove-source/{token}")
async def remove_source(token: str, sourcejson: SourceJson):
    decoded = check_auth(token)
    if not decoded:
        raise HTTPException(status_code=401, detail=RESPONSE_MESSAGES["invalid_auth"])
    email = decoded["email"]

    source = sourcejson.source

    db = get_db()
    cursor = db.cursor()
    cursor.execute("delete from sources where url = %s and email = %s", (source, email))
    db.commit()

    return {"message": f"Source {source} removed from database."}


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT"))
    host = "0.0.0.0"
    reload = bool(os.environ.get("IS_DEV"))
    uvicorn.run("server:app", host=host, port=port, reload=reload)
