# Public Table Datasets for LLMs & Table RAG 

Personal research notes on publicly available datasets for Table RAG, Table - - Question Answering LTable QAM, Text to SQL, and AI systems. 

# Goal 

- The purpose of this document is to collect useful table centric datasets that can be used for building and evaluating: 

- Table RAG 

- Table Question Answering 

- - 

- Text to SQL 

- AI Agents 

- Fact Verification 

- Enterprise Search 

- Financial AI 

Rather than keeping random bookmarks, this is a single place to compare datasets and understand where each one fits. 

# Dataset Categories 

Table Question Answering LTable QAM 

- - 

- Text to SQL 

- Fact Verification 

- Hybrid Table + Text QA 

- Enterprise Databases 

- Financial Datasets 

Untitled 

1 

# Useful Resources 

|Resource|Link|
|---|---|
|Hugging Face Datasets|https://huggingface.co/datasets|
|Papers With Code|https://paperswithcode.com/datasets|
|Kaggle|https://www.kaggle.com/datasets|
|UCI ML Repository|https://archive.ics.uci.edu/ml|
|OpenML|https://www.openml.org|
|Google Dataset Search|https://datasetsearch.research.google.com|



# Recommended Datasets 

|Dataset|Best For|Link|
|---|---|---|
|WikiTableQuestions|Table QA|https://github.com/ppasupat/WikiTableQuestions|
|WikiSQL|Beginner Text-<br>to-SQL|https://github.com/salesforce/WikiSQL|
|Spider|Advanced Text-<br>to-SQL|https://yale-lily.github.io/spider|
|BIRD|Enterprise Text-<br>to-SQL|https://bird-bench.github.io|
|TabFact|Fact Verification|https://tabfact.github.io|
|HybridQA|Table+Text QA|https://github.com/wenhuchen/HybridQA|
|FEVEROUS|Hybrid Retrieval|https://huggingface.co/datasets/feverous/feverous|
|SQA|Sequential<br>Table QA|https://github.com/microsoft/SQA|
|ToTTo|Table-to-Text<br>Generation|https://github.com/google-research-<br>datasets/ToTTo|
|TURL|Table<br>Understanding|https://github.com/sunlab-osu/TURL|
|FinQA|Financial QA|https://github.com/czyssrs/FinQA|



Untitled 

2 

|Dataset|Best For|Link|
|---|---|---|
|TAT]QA|Financial<br>Reasoning|https://nextplusplus.github.io/TAT]QA/|



# What I Learned 

After looking through these datasets, a few things stood out. 

- Most popular benchmarks are built using Wikipedia tables. 

- - 

- Spider and BIRD are currently the most widely used benchmarks for Text to SQL. 

- Enterprise systems usually need more than vector search; they combine SQL, metadata, and semantic retrieval. 

- Financial datasets focus much more on numerical reasoning than general QA datasets. 

- 

- Hybrid datasets like HybridQA and FEVEROUS are closer to real world applications because they require both tables and text. 

# Why Tables Are Challenging 

Unlike normal documents, tables store information through rows, columns, and relationships. 

Some common challenges are: 

- Understanding schemas 

- Numerical reasoning 

- Aggregation 

- Filtering 

- 

- Multi table joins 

- Preserving row-column relationships 

- Handling large databases 

Untitled 

3 

This is why many modern systems combine SQL execution with retrieval instead of relying only on embeddings. 

# Project Ideas 

Some interesting projects that could be built using these datasets: 

Build a Table RAG system over CSV or SQL databases. 

- Compare row-level vs table-level retrieval. 

- - - 

- Fine tune an LLM for Text to SQL using Spider. 

- Build an AI data analyst that converts natural language into SQL. 

- Create an enterprise chatbot that can answer questions from structured business data. 

# Recommended Datasets by Use Case 

## Learning Table QA 

- WikiTableQuestions 

- SQA 

## - - Learning Text to SQL 

- WikiSQL 

- Spider 

## Enterprise AI 

- BIRD 

- Spider 

## Table RAG 

- HybridQA 

Untitled 

4 

- FEVEROUS 

- TabFact 

## Financial AI 

- FinQA 

- ] 

- TAT QA 

# Next Steps 

Things I want to explore next: 

- Table chunking strategies 

- Table embeddings 

- Hybrid SQL � Vector Search 

- Schema retrieval 

- Enterprise Table RAG architectures 

- Recent research papers on structured retrieval 

# References 

- Hugging Face — https://huggingface.co/datasets 

- Papers With Code — https://paperswithcode.com/datasets 

- ACL Anthology — https://aclanthology.org 

- arXiv — https://arxiv.org 

- Stanford NLP — https://nlp.stanford.edu 

- Google Research — https://research.google 

Status: Ongoing research. I'll continue updating this page as I discover new datasets, papers, and production approaches. 

Untitled 

5 

