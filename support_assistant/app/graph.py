from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.retrieval import retrieve

class AgentState(TypedDict):
    
    query:str
    intent:str
    
    answer:str
    sources:list
    confidence:float
    
    
POLICY_KEYWORDS=[
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def classify_intent(state):

    query=state["query"].lower()

    if any(word in query for word in POLICY_KEYWORDS):
        
        state["intent"] = "policy_question"
        
    else:
        
        state["intent"] = "general_question"
        
    return state

def retrieve_and_answer(state):
    
    docs = retrieve(state["query"])
    
    state["answer"]=(
        "Based on retrieved context:" + docs[0].page_content[:200]
    )
    
    state["sources"]=[
        doc.metadata["source"]
        for doc in docs
    ]
    
    state["confidence"]=1.0
    
    return state

def direct_answer(state):
    state["answer"]=("I can only answer questions" "about Zepto policies.")
    
    state["sources"]=[]
    state["confidence"]=1.0
    
    return state

def router(state):
    
    return state["intent"]

builder = StateGraph(AgentState)

builder.add_node(
    "classify_intent",classify_intent)

builder.add_node(
    "retrieve_and_answer",retrieve_and_answer)

builder.add_node(
    "direct_answer",direct_answer)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",router,
    {
        "policy_question":"retrieve_and_answer",
        "general_question":"direct_answer"
    }
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()