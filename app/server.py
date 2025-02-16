import logging
from decouple import config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicRequest(BaseModel):
    topic: str

app = FastAPI()
model = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))
system_message = """
You are a teacher who is an expert at preparing test/quiz papers.
Ideal number of questions in a quiz is 5-10.
Create a quiz on the given topic in the following JSON structure:
{{
  "title": "Quiz title",
  "description": "Description of quiz",
  "creatorId": "dummy email",
  "creator": "dummy name",
  "timer": 20,
  "status": "public",
  "questions": [
    {{
      "query": {{
        "text": "Question text",
        "image": null
      }},
      "options": [
        {{
          "text": "Option1",
          "image": null
        }},
        {{
          "text": "Option2",
          "image": null
        }},
		{{
          "text": "Option3",
          "image": null
        }},
		{{
          "text": "Option4",
          "image": null
        }}
      ],
      "correctAnswer": "correct answer",
	  "explanation":"explanation of correct answer"
      "timer": 10
    }},
	...
	]
}}
"""

prompt = ChatPromptTemplate.from_template(
    "System: " + system_message + "\nUser: Create a quiz on topic '{topic}'.")
parser = JsonOutputParser()
chain = prompt | model| parser

@app.post("/openai")
async def create_quiz(request: TopicRequest):
    try:
        logger.info(f"Received request with topic: {request.topic}")
        # Create the prompt input
        prompt_input = {"topic": request.topic}
        # Generate the quiz using the chain
        result = chain.invoke(prompt_input)
        logger.info("Quiz generated successfully")       
        return result
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)