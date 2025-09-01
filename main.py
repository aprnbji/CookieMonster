from dependencies import *

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key
)

search = GoogleSerperAPIWrapper()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


chroma_client = chromadb.PersistentClient(path="./database")
cve_collection = chroma_client.get_collection("cve_database")

vectorstore = Chroma(
    client=chroma_client,
    collection_name="cve_database",
    embedding_function=embedding_model
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

all_results = cve_collection.get(include=["documents", "metadatas"])
docs = [
    Document(page_content=all_results["documents"][i], metadata=all_results["metadatas"][i])
    for i in range(len(all_results["documents"]))
]
sparse_retriever = BM25Retriever.from_documents(docs)

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

rerank_client = Client(base_url="http://localhost:7997")
reranker_model = "Alibaba-NLP/gte-multilingual-reranker-base"

compressor = InfinityRerank(client=rerank_client, model=reranker_model)
InfinityRerank.model_rebuild() 

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)


reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

@tool
def get_cve_info(query: str) -> str:
    """
    CVE Lookup Tool
    MUST be used for any query involving CVEs, vulnerabilities, exploits, 
    or security flaws. This includes named software, libraries, 
    or general attack descriptions.

    This tool searches the ChromaDB CVE database for relevant vulnerabilities
    and reranks results using InfinityRerank + CrossEncoder.

    Output Format:
    Title: <CVE title or ID>
    Severity: <CVSS severity or 'Unknown'>
    Summary: <Brief description of the vulnerability>
    """
    try:
        results = compression_retriever.invoke(query)
    except Exception:
        results = []

    if not results:
        results = hybrid_retriever.get_relevant_documents(query)

    if not results:
        return "No matching CVEs found in the database."

    output_lines = []
    for idx, doc in enumerate(results):
        title = doc.metadata.get("cve_id", doc.metadata.get("title", f"Doc-{idx+1}"))
        severity = doc.metadata.get("cvss_v2_severity", doc.metadata.get("severity", "Unknown"))
        summary = doc.page_content.split("CVSS")[0].strip()
        output_lines.append(f"{idx+1}. **{title}** (Severity: {severity})\n   {summary}")

    return "\n\n".join(output_lines)


@tool
def web_search(query: str) -> str:
    """ 
    Web Search Tool This tool performs a web search using the Google Serper API and returns the top result. 
    It is useful when a user asks for information that can be found online. 
    """
    results = search.run(query)
    return results if results else "No relevant results found."


prompt = """
You are CookieMonster, a helpful AI assistant.
You have access to tools that can retrieve information from external databases.

- Always first try to answer using your own knowledge.
- If the user’s query is relevant to a tool you have access to and you cannot fully or confidently answer from your own knowledge, then use the appropriate tool.
- Only use a tool when it will significantly improve accuracy or provide missing details.
- When using a tool, clearly present the result in the required output format.
"""

agent = create_react_agent(
    llm,
    tools=[get_cve_info],
    checkpointer=InMemorySaver(),
    prompt=prompt
)

@traceable(name="CookieMonsterSession")
def run_agent_conversation():
    print(pyfiglet.figlet_format("CookieMonster"))
    print("Hello. Hope you had an ugly day. Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("GoodbyE Ass")
            break
        
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={
                "configurable": {"thread_id": "1"},
                "metadata": {
                    "user": "cli_user",
                    "session_time": datetime.now().isoformat()
                }
            }
        )
        print("Response:", response["messages"][-1].content)


if __name__ == "__main__":
    run_agent_conversation()
