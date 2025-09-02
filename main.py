from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
import json
from tool_ToF import true_or_false
from tool_MC import multiple_choice

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt setup
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI quiz generator with two quiz creation tools:
            1. true_or_false - creates True/False questions
            2. multiple_choice - creates Multiple Choice questions with 4 options (A, B, C, D)
            
            The user will provide both the quiz type and the content in their message.
            
            Use the appropriate tool based on what the user requests:
            - If they mention "true/false", "T/F", or similar, use the true_or_false tool
            - If they mention "multiple choice", "MCQ", or similar, use the multiple_choice tool
            
            Extract the text content from their message and pass it to the appropriate tool.
            The content is usually after words like "about:", "on:", "from:", or similar.
            
            Save the quiz as a JSON file and provide confirmation when complete.
            Keep your responses brief and focused on the file creation confirmation.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [true_or_false, multiple_choice]

# AI AGENT
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def display_quiz_results():
    """Display results for both possible quiz types"""
    quiz_files = [
        ("quiz.json", "TRUE/FALSE QUIZ"),
        ("multiple_choice_quiz.json", "MULTIPLE CHOICE QUIZ")
    ]
    
    for filename, quiz_type in quiz_files:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
            
            print(f"\n{'='*60}")
            print(f"{quiz_type} SUCCESSFULLY CREATED!")
            print(f"{'='*60}")
            print(f"Number of questions: {quiz_data['number_of_questions']}")
            print(f"File saved as: {filename}")
            
            print(f"\n{'='*60}")
            print("PREVIEW OF QUESTIONS:")
            print(f"{'='*60}")
            
            for i, q in enumerate(quiz_data['questions'], 1):
                print(f"{i}. {q['question']}")
                
                if 'answer' in q:  # True/False format
                    print(f"   Answer: {q['answer']}")
                elif 'options' in q:  # Multiple Choice format
                    for j, option in enumerate(q['options']):
                        letter = chr(65 + j)  # Convert 0,1,2,3 to A,B,C,D
                        print(f"   {letter}. {option}")
                    print(f"   Correct Answer: {q['correct_answer']}")
                print()
            
            break  # Only display the quiz that was created
            
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            print(f"\nError: Could not read the {filename} file.")
            break
        except Exception as e:
            print(f"\nError reading {filename}: {e}")
            break
    else:
        print("\nNo quiz files found.")

def main():
    print("=== AI Quiz Generator ===")
    print("Available quiz types:")
    print("- True/False questions")
    print("- Multiple Choice questions (A, B, C, D)")
    print()
    print("Instructions:")
    print("1. Specify the quiz type (true/false or multiple choice)")
    print("2. Provide the text content you want to create questions from")
    print("Example: 'Create a multiple choice quiz about: Biology is the study of life...'")
    print()
    
    # User input
    query = input("Enter quiz type and content: ")
    
    # Run the agent
    response = agent_executor.invoke({"query": query})
    print(f"\n{response['output']}")
    
    # Display the created quiz
    display_quiz_results()

if __name__ == "__main__":
    main()