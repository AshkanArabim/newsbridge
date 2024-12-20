from abc import ABC, abstractmethod
import ollama
import subprocess
import time
import google.generativeai as genai
import re


class AbstractModel(ABC):
    def __init__(self):
        # match all punctuation followed by whitespace
        # src: https://www.freecodecamp.org/news/what-is-punct-in-regex-how-to-match-all-punctuation-marks-in-regular-expressions/
        # I can't hardcode punctuation characters since the output won't
        # just be in English.
        # TODO: find a way to at least ignore commas and quotes in languages
        self.delimiter = r"[^\w\s]+"

    @abstractmethod
    def generate(self, prompt: str):
        """
        Generates a streaming response for the given prompt.

        Yields one phrase at a time. A phrase is a sequence of words ending with
        a punctuation mark.
        """
        pass


class Llama(AbstractModel):
    def __init__(self, llama_name: str):
        super().__init__()

        # src: https://stackoverflow.com/a/78501628/14751074
        # translated the answer's bash script logic to python

        # start the ollama server in the background
        print("Waiting for Ollama server to start")
        subprocess.Popen(
            ["ollama", "serve"],
            start_new_session=True,
        )
        time.sleep(0.5)  # ugly hack.

        # download the model if it's not here
        print("Downloading if'", llama_name, "'not already downloaded...", flush=True)
        ollama.pull(llama_name)

        self.model_name = llama_name

    def generate(self, prompt: str):
        stream = ollama.generate(model=self.model_name, prompt=prompt, stream=True)

        # llama outputs mostly in words, so I'm piecing them together and
        # returning when a phrase is complete.
        phrase_word_list = []
        for chunk_obj in stream:
            chunk = chunk_obj["response"]
            phrase_word_list.append(chunk)

            if re.search(self.delimiter, chunk):
                phrase = "".join(phrase_word_list)
                phrase_word_list = []
                yield phrase

        # if no more responses, return whatever's left
        if phrase_word_list:
            yield "".join(phrase_word_list)


class Gemini2(AbstractModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__()

        # src: copied code from Google's AI studio
        genai.configure(api_key=api_key)

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

    def generate(self, prompt: str):
        # src: https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md#generate_content
        stream = self.model.generate_content(prompt, stream=True)

        # gemini's phrase size is between one sentence and one paragraph.
        # I'll have to split the content into words and iterate over them
        phrase_word_list = []
        for chunk_obj in stream:
            chunk = chunk_obj.text

            # split phrase into words (including the whitespace) & append to list
            space_delimiter = r"\s"
            whitespaces = re.findall(space_delimiter, chunk)
            words = re.split(space_delimiter, chunk)
            for i in range(len(whitespaces)):
                words[i] += whitespaces[i]

            # iterate over the words, yield when a phrase is completed
            for word in words:

                phrase_word_list.append(word)
                if re.search(self.delimiter, word):
                    phrase = "".join(phrase_word_list)
                    phrase_word_list = []
                    yield phrase

        # if no more responses, return whatever's left
        if phrase_word_list:
            yield "".join(phrase_word_list)
