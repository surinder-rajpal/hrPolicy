from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_confidence_agent(llm): 
    confidence_prompt = PromptTemplate.from_template(
        """
        You are an expert in generating the confidence score and risk. Analyze the query, context and decision.
        Provide the confidence on scale of 0-1 and risk value low/medium/high. Provide the json output.
        
        Query: {query}
        Context: {context}
        decision: {decision}
        
        Output:
            result:
                confidenc: "",
                risk: ""
        """
    )
    confidence_agent = confidence_prompt | llm | JsonOutputParser()
    return confidence_agent