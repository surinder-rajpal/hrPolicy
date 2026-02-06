from policy_loader import get_retriever

def retrieve_docs(query):
    
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])