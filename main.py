from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time, os

import warnings
warnings.filterwarnings("ignore")


from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

template = """
1. Input Detection:
First, determine the type of input:
If the input is a link (URL):
Proceed to the next step to analyze the webpage.
If the input is a piece of text (not a URL):
Proceed to the next step to analyze the text.
If the input doesn't match either of these, it is invalid.
Output: "Invalid output."

2. If the input is a URL (webpage):
Open the URL and read through the content of the page.
Identify terms and conditions or any legal clauses that might exist within the webpage.
Categorize each term:
High-risk terms:
Identify extreme or critical terms that could have serious legal, financial, or personal consequences if agreed to.
These terms should be flagged as high-risk, and the user should be warned to consult legal or professional advice before agreeing to them.
Moderate-risk terms:
Identify terms that are important but not as severe. These should be flagged as moderate risk, with a recommendation to read carefully and understand the implications.
Return the categorized high-risk and moderate-risk terms.

3. If the input is text (not a URL):
Analyze the text and briefly explain it in no more than 50 words.
Summarize the key idea or implications of the text without elaborating on irrelevant details.
4. If neither a link nor a piece of text is entered:

Output: "Invalid output."
"""


google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")



from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            template
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=google_llm,
    prompt=prompt,
    memory=memory
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or '*' to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def get_response(request: ChatRequest):
    """
    Chat endpoint for generating a response based on the prompt and context.
    """
    try:
        start = time.process_time()
        response = conversation.invoke(request.prompt)
        print("Response time: ", time.process_time() - start)
        return ChatResponse(
            answer=response["text"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)