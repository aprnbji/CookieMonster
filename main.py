from dependencies import *

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.2
)

search = GoogleSerperAPIWrapper()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")


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
sparse_retriever.k = 10

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

def expand_query(query: str) -> dict:
    prompt = f"""
        You are an expert query optimizer. Your task is to expand a user's query into three distinct versions for a Retrieval-Augmented Generation (RAG) system.

        The user query is: "{query}"

        Generate three versions:
        1.  **original**: The query exactly as it was provided.
        2.  **broad**: A more generalized version that captures the high-level topic. Remove specific details and focus on the core concept.
        3.  **specific**: A more detailed version that adds context or focuses on a specific, searchable entity within the query.

        Return the result as a single, valid JSON object with the keys "original", "broad", and "specific".

        Example:
        User Query: "log4j vulnerability"
        {
            "original": "log4j vulnerability",
            "broad": "java library security vulnerabilities",
            "specific": "CVE details for Apache Log4j remote code execution flaw"
        }
        """
    try:
        response = llm.invoke(prompt)
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        expanded = json.loads(cleaned_response)
    except (json.JSONDecodeError, Exception) as e:
        # print(f"Warning: Failed to expand query with LLM. Falling back to original. Error: {e}")
        expanded = {"original": query, "broad": query, "specific": query}
    
    # print(f"Expanded Queries: {expanded}")
    return expanded

def ensemble_retrieve(query: str, top_k: int = 5) -> list[Document]:
    # print(f"\n--- Starting Advanced Retrieval for: '{query}' ---")
    
    expanded_queries = expand_query(query)
    
    candidate_docs = []
    seen_contents = set()

    for q_type, q_text in expanded_queries.items():
        retrieved_docs = hybrid_retriever.invoke(q_text)
        
        if retrieved_docs:
            sentence_pairs = [[q_text, doc.page_content] for doc in retrieved_docs]
            scores = reranker.predict(sentence_pairs)
            
            reranked_pairs = list(zip(scores, retrieved_docs))
            reranked_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # print(f"Found {len(reranked_pairs)} candidates for '{q_type}' variant.")

            for score, doc in reranked_pairs[:5]:  # Take top 5 from each variant
                if doc.page_content not in seen_contents:
                    doc.metadata["variant"] = q_type
                    doc.metadata["local_score"] = float(score)
                    candidate_docs.append(doc)
                    seen_contents.add(doc.page_content)

    # print(f"\nTotal unique candidates after local reranking: {len(candidate_docs)}")

    if not candidate_docs:
        return []

    global_sentence_pairs = [[query, doc.page_content] for doc in candidate_docs]
    global_scores = reranker.predict(global_sentence_pairs)
    
    final_reranked_pairs = list(zip(global_scores, candidate_docs))
    final_reranked_pairs.sort(key=lambda x: x[0], reverse=True)
    
    final_docs = [doc for score, doc in final_reranked_pairs]
    # print("--- Advanced Retrieval Finished ---")
    
    return final_docs[:top_k]


@tool
def get_cve_info(query: str) -> str:
    """
    CVE Lookup Tool
    Must be used for any query involving CVEs, vulnerabilities, exploits, 
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
        results = ensemble_retrieve(query, top_k=5)
    except Exception as e:
        return f"Error during retrieval: {e}"

    if not results:
        return "No matching CVEs found in the database."

    output_lines = []
    for idx, doc in enumerate(results):
        title = doc.metadata.get("cve_id", doc.metadata.get("title", f"Doc-{idx+1}"))
        severity = doc.metadata.get("cvss_v2_severity", doc.metadata.get("severity", "Unknown"))
        summary = doc.page_content.split("CVSS")[0].strip()
        variant = doc.metadata.get("variant", "unknown")

        
        output_lines.append(
            f"{idx+1}. **{title}** (Severity: {severity})\n   {summary}"
        )

    return "\n\n".join(output_lines)


@tool
def web_search(query: str) -> str:
    """ 
    Web Search Tool 
    This tool performs a web search using the Google Serper API and returns the top result. 
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
