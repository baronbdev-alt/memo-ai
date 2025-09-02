import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Step 1: Define Pydantic schema
class Question(BaseModel):
    question: str
    answer: bool

class QuizResponse(BaseModel):
    input_text: str
    number_of_questions: int
    questions: List[Question]

# Step 2: Setup parser
parser = PydanticOutputParser(pydantic_object=QuizResponse)

# Step 3: Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quiz generator. Create true/false questions from the given text."),
    ("user", """Text to generate quiz from:
{text}
     
Number of questions: {num_questions} 

{format_instructions}
""")
])

# Step 4: Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 5: Define function that saves to JSON and returns confirmation
def generate_quiz(text: str, num_questions: int = 4, output_file: str = "quiz.json"):
    # Run chain
    chain = prompt | llm | parser
    response: QuizResponse = chain.invoke({
        "text": text,
        "num_questions": num_questions,  # <-- Pass this!
        "format_instructions": parser.get_format_instructions()
    })

    # Convert Pydantic object to dict
    response_dict = response.dict()

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)

    return f"Quiz with {len(response.questions)} questions saved to {output_file}"

true_or_false = Tool(
    name="generate_a_true_or_false_quiz",
    func=generate_quiz,
    description="Creates a True or False quiz from the given text and saves it as a JSON file. Returns confirmation when complete.",
)