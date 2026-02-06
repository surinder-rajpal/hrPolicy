from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_case_analysis_agent(llm): 
    case_analysis_prompt = PromptTemplate.from_template(
        """
        You are an expert HR case analyst. Fetch the key information from the query and provide if 
        missing any key information required for the case. Do not consider company name or employee id.
        If the intent is lookup, then just consider the topic. Provide the json output.
        
        Query: {query}
        Intent: {intent}
        
        Output:
            key_info: [key: value]
            missing_info: [key]
        """
    )
    case_analysis_agent = case_analysis_prompt | llm | JsonOutputParser()
    return case_analysis_agent