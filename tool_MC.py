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

# Step 1: Define Pydantic schema for multiple choice
class MultipleChoiceQuestion(BaseModel):
    question: str
    options: List[str]  # List of 4 options (A, B, C, D)
    correct_answer: str  # The letter of the correct answer (A, B, C, or D)

class MultipleChoiceQuizResponse(BaseModel):
    input_text: str
    number_of_questions: int
    questions: List[MultipleChoiceQuestion]

# Step 2: Setup parser
parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuizResponse)

# Step 3: Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quiz generator. Create multiple choice questions with 4 options (A, B, C, D) from the given text. Make sure to provide exactly 4 options for each question and specify the correct answer as A, B, C, or D."),
    ("user", """Text to generate quiz from:
{text}
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
def generate_multiple_choice_quiz(text: str, num_questions: int = 5, output_file: str = "quiz.json"):
    # Run chain
    chain = prompt | llm | parser
    response: MultipleChoiceQuizResponse = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

    # Convert Pydantic object to dict
    response_dict = response.dict()

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)

    # Return simple confirmation message
    return f"Multiple choice quiz with {len(response.questions)} questions saved to {output_file}"

multiple_choice = Tool(
    name="generate_a_multiple_choice_quiz",
    func=generate_multiple_choice_quiz,
    description="Creates a Multiple Choice quiz from the given text with 4 options (A, B, C, D) for each question and saves it as a JSON file. Returns confirmation when complete.",
)