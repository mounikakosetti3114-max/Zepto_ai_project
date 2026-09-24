
This project is a capstone project for Zepto. It has **three parts**
(called modules), and all three live together in this one repository.

## Folder structure

```
Zepto_ai_project/
├── data_pipeline/        # Module 1 — collects and cleans data
├── analytics/             # Module 2 — data analysis and ML models
├── support_assistant/     # Module 3 — the customer support chatbot
└── README.md              # this file
```

Each folder above has its **own README.md** with all the details for
that module — exact steps, results, explanations, everything. This file
you're reading now is just a simple overview and a starting point.

---

## Before you start: install the requirements

Each module has its own `requirements.txt` file listing the Python
packages it needs (they're different for each module, so install them
separately):

```bash
cd data_pipeline
pip install -r requirements.txt

cd ../analytics
pip install -r requirements.txt

cd ../support_assistant
pip install -r requirements.txt
```

---

## How to run each module

### 1️⃣ Module 1 — Data Pipeline
This part scrapes book data from a practice website, cleans it up, and
saves it into a small database.

```bash
cd data_pipeline
pip install -r requirements.txt
python pipeline.py
```

Full details: [`data_pipeline/README.md`](data_pipeline/README.md)

### 2️⃣ Module 2 — Analytics Pipeline
This part loads the Titanic dataset, explores it, cleans it, and then
trains a few machine learning models to predict survival.

```bash
cd analytics
pip install -r requirements.txt
python 01_eda.py        # step 1: explore and clean the data
python 02_modeling.py   # step 2: train and evaluate the models
```

Run `01_eda.py` first — it saves a `titanic.csv` file that
`02_modeling.py` needs. Full details, results, and the final
recommendation: [`analytics/README.md`](analytics/README.md)

### 3️⃣ Module 3 — Support Assistant
This part is a simple chatbot. You ask it a question about Zepto's
policies (like delivery time or returns), and it looks through Zepto's
policy documents to answer you.

```bash
cd support_assistant
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt
python app/ingest.py                        # step 1: loads the documents
uvicorn app.main:app --reload --port 7860   # step 2: starts the chatbot server
```

Full details and example questions: [`support_assistant/README.md`](support_assistant/README.md)

---

## What's inside each module (short summary)

**Module 1 — Data Pipeline:** Scrapes book listings from a website,
cleans up the price, rating, and stock information, converts prices to
INR, and stores everything neatly in a small SQLite database.

**Module 2 — Analytics Pipeline:** Works with the famous Titanic
dataset. It fills in missing information sensibly (for example, the
`age` column was missing about 20% of the time, so we filled those gaps
in with the median age), explores which passengers were more likely to
survive, and then trains three different models to predict survival.
After comparing them, the **Random Forest model** performed best and is
the one we'd recommend using.

**Module 3 — Support Assistant:** A small chatbot for Zepto customer
support. It reads 8 policy documents (about delivery, returns,
membership, etc.), and when you ask it something, it searches those
documents for the most relevant answer. It's built with LangGraph
(which decides what to do with each question) and ChromaDB (which
stores and searches the documents).

---

## About the code history

If you look at the commit history of this project (`git log`), you'll
see a feature branch that was created, had a couple of commits, and was
then merged back into the main branch — this is normal software
development practice.

## A note on cost

Everything in this project runs for free — no paid subscriptions, no
credit card needed anywhere. Any optional "real AI" mode in Module 3
only uses free-tier services if you choose to try it.