from dependencies import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CookieMonster")

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
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
        You are an expert query optimizer. Expand a user's query into three distinct versions for a RAG system.

        User query: "{query}"

        Generate three versions:
        1. original: The exact query.
        2. broad: A generalized version.
        3. specific: A more detailed contextual version.

        Return only valid JSON with keys "original", "broad", and "specific".
    """
    response = llm.invoke(prompt)
    cleaned = response.content.strip()
    if not cleaned.startswith("{"):
        cleaned = cleaned[cleaned.find("{"):]
    if not cleaned.endswith("}"):
        cleaned = cleaned[:cleaned.rfind("}")+1]
    expanded = json.loads(cleaned)

    print("Expanded queries:", expanded)

    return expanded


def ensemble_retrieve(query: str, top_k: int = 5) -> list[Document]:
    expanded_queries = expand_query(query)

    candidate_docs = []
    seen_ids = set()

    # Collect all docs from all expanded queries
    for q_type, q_text in expanded_queries.items():
        try:
            retrieved_docs = hybrid_retriever.get_relevant_documents(q_text)
        except Exception:
            continue

        for doc in retrieved_docs:
            cve_id = doc.metadata.get("cve_id", doc.metadata.get("title", None))
            if cve_id and cve_id not in seen_ids:
                doc.metadata["variant"] = q_type
                candidate_docs.append(doc)
                seen_ids.add(cve_id)

    if not candidate_docs:
        return []

    # One global rerank
    sentence_pairs = [[query, doc.page_content] for doc in candidate_docs]
    global_scores = reranker.predict(sentence_pairs)

    reranked_pairs = list(zip(global_scores, candidate_docs))
    reranked_pairs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in reranked_pairs][:top_k]

@tool
def analyze_logs(path_or_text: str) -> str:
    """
    Analyze plaintext logs (syslog, journalctl, app logs).
    Accepts either a file path or raw log text.
    """

    lines = []
    if os.path.exists(path_or_text):
        with open(path_or_text, "r", errors="ignore") as f:
            lines = f.readlines()
    else:
        lines = path_or_text.split("\n")

    config = TemplateMinerConfig()
    persistence = FilePersistence("log.bin")
    miner = TemplateMiner(persistence, config=config)

    clusters = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        res = miner.add_log_message(line)
        if res is None:  
            continue
        cid = res["cluster_id"]
        tpl = res["template_mined"]
        clusters.setdefault(cid, {"template": tpl, "count": 0})
        clusters[cid]["count"] += 1

    if not clusters:
        return "No significant log patterns found."

    result = "Log Patterns:\n"
    for idx, (cid, info) in enumerate(
        sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True)[:5], 1
    ):
        result += f"{idx}. [{info['count']} occurrences] {info['template']}\n"
    return result

@tool
def get_cve_info(query: str) -> list[dict]:
    """
    CVE Lookup Tool
    Returns structured JSON with CVE metadata for the given query.
    """
    
    try:
        results = ensemble_retrieve(query, top_k=5)
    except Exception as e:
        return [{"error": f"Retrieval failed: {str(e)}"}]

    if not results:
        return []

    structured_results = []
    for idx, doc in enumerate(results):
        structured_results.append({
            "cve_id": doc.metadata.get("cve_id", doc.metadata.get("title", f"Doc-{idx+1}")),
            "severity": doc.metadata.get("cvss_v2_severity", doc.metadata.get("severity", "Unknown")),
            "summary": doc.page_content.split("CVSS")[0].strip(),
            "variant": doc.metadata.get("variant", "unknown")
        })

    return structured_results

@tool
def web_search(query: str) -> str:
    """Web Search Tool using Google Serper API"""
    results = search.run(query)
    return results if results else "No relevant results found."


prompt = """
You are CookieMonster, a helpful AI assistant.
You have access to tools that can retrieve information from external databases.

Rules:
- Always first try to answer using your own knowledge.
- If the user’s query is related to CVEs, vulnerabilities, exploits, or security flaws, use the `get_cve_info` tool.
- The tool returns structured JSON. Your job is to format this JSON into a **numbered list**.
- For each CVE, show:
    1. CVE ID
    2. Severity
    3. Summary (1–2 sentences max)
- If no CVEs are found, say: "I couldn’t find any matching CVEs."
- If a JSON object has an "error" key, display the error message to the user.
- Never just dump raw JSON — always present the list in plain text.
"""

agent = create_react_agent(
    llm,
    tools=[get_cve_info, web_search, analyze_logs],
    checkpointer=InMemorySaver(),
    prompt=prompt
)


def run_agent_conversation():
    print(pyfiglet.figlet_format("CookieMonster"))
    print("Hello. Hope you had an ugly day. Type 'exit' or 'quit' to end the conversation.")

    history = []

    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("GoodbyE Ass")
            break

        try:
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
            reply = response["messages"][-1].content
        except Exception as e:
            reply = f"Agent failed: {e}"

        print("Response:", reply)
        history.append({"user": query, "agent": reply})

    return history

if __name__ == "__main__":
    run_agent_conversation()
