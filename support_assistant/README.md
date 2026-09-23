# Module 3 – Support Assistant

This project is a simple **Zepto Customer Support Assistant**.

It uses:
- Python
- LangGraph
- ChromaDB
- Sentence Transformers
- `all-MiniLM-L6-v2`

The project works in **offline/mock mode**, so an LLM API key is not required.

## 1. What does this project do?

The assistant answers questions using 8 Zepto policy documents.

Example:

```text
User: What is the delivery fee for an order below INR 149?

Assistant: Orders below INR 149 have a flat INR 25 delivery fee.
```

The application first checks whether the question is related to a Zepto policy. If it is, it searches the documents and uses the relevant information to answer.

## 2. Documents

The `docs` folder contains these 8 files:

```text
docs/
├── doc_01.txt   # Delivery Policy
├── doc_02.txt   # Returns & Refunds
├── doc_03.txt   # Membership Tiers
├── doc_04.txt   # Order Tracking
├── doc_05.txt   # Order Cancellation
├── doc_06.txt   # Damaged or Missing Items
├── doc_07.txt   # Gift Cards
└── doc_08.txt   # Customer Support Hours
```

The document text should be kept exactly as provided in the assignment.

## 3. How it works

The basic flow is:

```text
User Question
      ↓
Check Question Type
      ↓
Policy Question?
   /           Yes           No
  ↓             ↓
Search Docs   General Answer
  ↓
Get Top 3 Results
  ↓
Create Answer
```

## 4. Intent Classification

In the required mock mode, the application checks for these words:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If the question contains one of these words, it is treated as a:

```text
policy_question
```

Otherwise it is treated as:

```text
general_question
```

## 5. Embeddings and ChromaDB

The project uses:

```text
all-MiniLM-L6-v2
```

to convert text into numerical vectors called **embeddings**.

These embeddings are stored in **ChromaDB**.

When a user asks a policy question:

1. The question is converted into an embedding.
2. ChromaDB searches for similar documents.
3. The top 3 results are selected.
4. These results are used to create the answer.

## 6. LangGraph

LangGraph controls the steps of the application.

The project has at least 3 nodes:

```text
classify_intent
retrieve_and_answer
general_answer
```

The graph decides which step should run based on the question.

## 7. Prompt

The answer prompt follows this simple structure:

```text
Role
Context
Task
Format
Length
```

It also contains an important rule:

```text
Do not answer using information that is not present
in the provided context.
```

A small example is also included in the prompt to show the expected answer format.

## 8. Mock LLM

The assignment requires the application to work without an online LLM.

By default:

```text
MOCK_LLM=1
```

or if `MOCK_LLM` is not set, the application should use mock mode.

This means:

- No API key is required.
- No LLM account is required.
- No network connection to an LLM provider is required.

The optional real-LLM mode can be enabled with:

```text
MOCK_LLM=0
```

## 9. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```



## 10. Example Questions

### Policy question

```text
What is the refund policy for damaged items?
```

The application searches ChromaDB and answers using the policy documents.

### Another policy question

```text
How can I track my order?
```

The application retrieves information from the Order Tracking document.

### General question

```text
What is the capital of India?
```

This is treated as a general question and does not require Zepto policy retrieval.

## 11. Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── src/
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── prompts.py
│   ├── graph.py
│   └── main.py
│
├── requirements.txt
└── README.md
```



## Summary

This project is a simple **RAG (Retrieval-Augmented Generation)** application.

```text
Documents
    ↓
Embeddings
    ↓
ChromaDB
    ↓
User Question
    ↓
LangGraph
    ↓
Relevant Context
    ↓
Answer
```

The main goal is to provide accurate answers using the information available in the Zepto policy documents.
