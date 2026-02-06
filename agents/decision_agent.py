from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_decision_agent(llm): 
    decision_prompt = PromptTemplate.from_template(
        """
        You are an expert HR case analyst. Analyze the query, key info and context. Provide the realistic decision
        and provide reason why that decion is made.
        If there is no info, then error out with the reason. Provide the json output.
        
        Query: {query}
        Context: {context}
        key_info: {key_info}
        
        Output:
            result:
                decision: "",
                reasoning: ""
            error: ""
        """
    )
    decision_agent = decision_prompt | llm | JsonOutputParser()
    return decision_agent