from langchain.agents import create_agent
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def intent_agent():
    model = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.2)
    agent = create_agent(model, 
                         system_prompt="""You are an expert HR policy intent identification agent. 
                            Identify the intent, whether it is just an policy lookup or case analysis query." 
                            Return the Intent.""")
    return agent