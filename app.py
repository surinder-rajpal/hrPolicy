import streamlit as st
from policy_loader import loadAndSplitPoilicies
from dotenv import load_dotenv
from agents import intent_agent, rag_agent, case_agent, confidence_agent, decision_agent
from langchain.chat_models import init_chat_model
st.title("HR Policy Agent")
path = 'policies/'

load_dotenv()
vectorDb = loadAndSplitPoilicies(path)
llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.2)
intent = intent_agent.intent_agent()
case_a = case_agent.get_case_analysis_agent(llm)
confidence = confidence_agent.get_confidence_agent(llm)
decision = decision_agent.get_decision_agent(llm)

input = st.chat_input("Your Query: ")
result = {}
if input is not None:
    progress_bar = st.progress(0, "Checking the intent")
    result = intent.invoke(
        {"messages": [{"role": "user", "content": input}]}
    )
    intent = result["messages"][-1].content
    result['intent'] =intent

    progress_bar.progress(20, "Finding in RAG store")
    context = rag_agent.retrieve_docs(input)
    result['context'] = context

    progress_bar.progress(50, "Analysing Case")
    case_analysis = case_a.invoke({'query': input, 'intent': intent})
    result['case_analysis'] = case_analysis
    # st.info(case_analysis)

    progress_bar.progress(80, "Making a decision")
    decision_result = decision.invoke({'query': input, 'context': context, 'key_info': case_analysis['key_info']})
    result['decision'] = decision_result

    progress_bar.progress(90, "Checking the result confidence and risk")
    confidence_result = confidence.invoke({'query': input, 'context': context, 'decision': decision_result['result']['decision']})
    progress_bar.progress(100, "Analysis Done")
    result['confidence'] = confidence_result
    # st.info(result)
    col1, col2 = st.columns([2, 1])
            
    with col1:
        st.subheader("Decision")
        st.info(result['decision']['result']['decision'])
        
    with col2:
        st.subheader("Confidence and Risk")
        st.warning(result['confidence'])
        
    with st.expander("Reasoning"):
        st.write(f"**Intent:** {result['intent']}")
        st.write(f"**Reasoning:** {result['decision']['result']['reasoning']}")
        st.markdown(f"** Context:**\n> {result['context']}")

