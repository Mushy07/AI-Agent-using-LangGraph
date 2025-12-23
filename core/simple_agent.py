import re
import os
from typing import TypedDict, List

from langgraph.graph import StateGraph, END


# =========================
# State Definition
# =========================

class AgentState(TypedDict):
    conversation_history: List[str]
    current_query: str
    retrieved_sources: List[str]
    final_answer: str
    error: str


# =========================
# Utility Functions
# =========================

def tokenize(text: str) -> set[str]:
    # Normalize text into lowercase word tokens.
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


# =========================
# Source Loading
# =========================

def load_sources(file_path=None) -> List[str]:
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "sources", "sources.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split("\n---\n")
    except FileNotFoundError:
        return []


SOURCES = load_sources()


# =========================
# Tool: Soft Search
# =========================

def search_sources(query: str) -> List[str]:
    query_tokens = tokenize(query)
    matches = []

    for block in SOURCES:
        block_tokens = tokenize(block)

        if query_tokens & block_tokens:
            matches.append(block.strip())

    return matches


# =========================
# Tool: Source Summarizer
# =========================

def summarize_sources(sources: List[str]) -> dict:
    # Analyze and summarize retrieved sources.
    summary = {
        "total_sources": len(sources),
        "url_sources": 0,
        "title_sources": 0,
        "topics": set()
    }
    
    for src in sources:
        if "URL:" in src:
            summary["url_sources"] += 1
        if "TITLE:" in src:
            summary["title_sources"] += 1
        
        # Extract topic keywords (simple word extraction)
        words = tokenize(src)
        # Filter out common words and keep meaningful ones
        meaningful_words = {w for w in words if len(w) > 4}
        summary["topics"].update(list(meaningful_words)[:3])
    
    return summary


# =========================
# Graph Nodes
# =========================

def planner_node(state: AgentState) -> AgentState:
    # Decide whether we should search or respond with an error.
    if not state["current_query"].strip():
        state["error"] = "Empty query received."
    else:
        state["error"] = ""
    return state


def researcher_node(state: AgentState) -> AgentState:
    # Retrieve relevant sources.
    try:
        sources = search_sources(state["current_query"])
        state["retrieved_sources"] = sources        
        # Use second tool to analyze sources
        if sources:
            summary = summarize_sources(sources)
            print(f"\n[Tool: summarize_sources] Found {summary['total_sources']} sources "
                  f"({summary['url_sources']} URLs, {summary['title_sources']} documents)")    
    except Exception as e:
        state["error"] = f"Search error: {str(e)}"
        state["retrieved_sources"] = []

    return state


def responder_node(state: AgentState) -> AgentState:
    # Generate final response based on retrieved sources.
    query = state["current_query"]

    if state["error"]:
        state["final_answer"] = f"Error: {state['error']}"
        return state

    if not state["retrieved_sources"]:
        state["final_answer"] = (
            f"No relevant sources found for: '{query}'.\n"
            "Try rephrasing your question."
        )
        return state

    answer = f"Answer based on sources for: '{query}'\n\n"
    answer += "=== CONTENT ===\n\n"
    
    references = []
    
    for i, src in enumerate(state["retrieved_sources"], 1):
        lines = src.strip().split("\n")
        content = ""
        source_ref = ""
        
        for line in lines:
            if line.startswith("CONTENT:"):
                content = line.replace("CONTENT:", "").strip()
            elif line.startswith("URL:"):
                source_ref = line.replace("URL:", "").strip()
            elif line.startswith("TITLE:"):
                source_ref = line.replace("TITLE:", "").strip()
        
        if content:
            answer += f"[{i}] {content}\n\n"
        
        if source_ref:
            references.append(f"[{i}] {source_ref}")
    
    if references:
        answer += "\n=== REFERENCES ===\n"
        answer += "\n".join(references)
    
    state["final_answer"] = answer
    return state


# =========================
# Conditional Logic
# =========================

def has_error(state: AgentState) -> str:
    return "error" if state["error"] else "ok"


# =========================
# Build LangGraph
# =========================

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("responder", responder_node)

graph.set_entry_point("planner")

graph.add_conditional_edges(
    "planner",
    has_error,
    {
        "error": "responder",
        "ok": "researcher"
    }
)

graph.add_edge("researcher", "responder")
graph.add_edge("responder", END)

agent = graph.compile()


# =========================
# Interactive Conversation Loop
# =========================

def run_agent():
    state: AgentState = {
        "conversation_history": [],
        "current_query": "",
        "retrieved_sources": [],
        "final_answer": "",
        "error": ""
    }

    print("Research Assistant Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Conversation ended.")
            break

        state["current_query"] = user_input

        # Run graph
        result = agent.invoke(state)

        # Print response (ONLY current answer)
        print("\nAgent:\n")
        print(result["final_answer"])
        print("\n" + "-" * 50 + "\n")

        # Print current state to show persistence (BEFORE clearing)
        print("=== CURRENT STATE ===")
        print(f"Conversation History: {result['conversation_history']}")
        print(f"Current Query: {result['current_query']}")
        print(f"Retrieved Sources Count: {len(result['retrieved_sources'])}")
        print(f"Error: {result['error'] if result['error'] else 'None'}")
        print("\n" + "-" * 50 + "\n")

        # Maintain memory (but not reuse output)
        state["conversation_history"].append(user_input)
        state["retrieved_sources"] = []
        state["final_answer"] = ""
        state["error"] = ""


if __name__ == "__main__":
    run_agent()
