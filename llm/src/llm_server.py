from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import llm_classes


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup and shutdown events
    # see https://fastapi.tiangolo.com/advanced/events/

    global model

    # startup
    # get model name
    model_name = os.environ.get("MODEL_NAME")
    assert model_name is not None, "MODEL_NAME env variable not set!"

    if model_name.startswith("llama"):
        model = llm_classes.Llama(llama_name=model_name)
    elif model_name.startswith("gemini"):
        gemini_model_name, api_key = model_name.split("|")

        model = llm_classes.Gemini2(model_name=gemini_model_name, api_key=api_key)
    else:
        raise Exception("Invalid model name:", model_name)

    # app starts here
    yield

    # shutdown
    # empty for now...


app = FastAPI(lifespan=lifespan)


@app.get("/api/generate")
def generate(prompt: str):
    def generate_stream():
        for phrase in model.generate(prompt):
            phrase = phrase.encode()
            yield phrase

    return StreamingResponse(generate_stream())


if __name__ == "__main__":
    port = os.environ.get("LLM_PORT")
    assert port is not None, "LLM_PORT in llm-server is not set!"
    port = int(port)

    is_dev = os.environ.get("IS_DEV", "false").lower() == "true"

    uvicorn.run(__name__ + ":app", host="0.0.0.0", port=port, reload=is_dev)
