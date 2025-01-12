

# Retrieval-Augmented Generation (RAG) Validation

## Overview

This project focuses on validating a **Retrieval-Augmented Generation (RAG)** pipeline by assessing its retrieval and generation capabilities using specific metrics like **Recall**, **Precision**, **F1-Score**, and qualitative evaluations of **Answer Relevance** and **Faithfulness**.

## RAG Pipeline

The pipeline follows these steps:

1. **Document Chunking**  
   - Split the document into smaller chunks of fixed size.
   - Assign a `chunk_id` to each chunk for easy identification.

2. **Q&A Generation**  
   - Use a **Large Language Model (LLM)** to generate **questions and answers (Q&A)** for each chunk.
   - These Q&A pairs will serve as the basis for retrieval and evaluation.

3. **Embedding and ElasticSearch**  
   - Embed the chunks into a vector space for efficient similarity search.
   - Perform retrieval using **ElasticSearch** based on the generated questions.

4. **Retrieve Relevant Chunks**  
   - Retrieve the top-k most relevant chunks from ElasticSearch.

5. **RAG Answer Generation**  
   - Use the retrieved chunks as input to the LLM to generate the final answer.

6. **Evaluation**  
   - Evaluate the system’s performance using the following metrics:
     - **Recall**: Proportion of relevant chunks retrieved.
     - **Precision**: Accuracy of retrieved chunks (evaluated for specific `k` values).
     - **F1-Score**: Balance between Recall and Precision.
     - **Answer Relevance**: Does the generated answer align with the question?
     - **Faithfulness**: Is the generated answer consistent with the retrieved chunks?

---

## Pipeline Details

### Step 1: Document Chunking
- **Objective**: Divide documents into smaller, manageable units for retrieval.  
- **Implementation**:
  - Split documents into chunks of fixed size (e.g., 200 tokens).
  - Assign a unique `chunk_id` to each chunk.  

---

### Step 2: Q&A Generation
- **Objective**: Create synthetic Q&A pairs to test retrieval and answer generation.  
- **Implementation**:
  - Use an LLM to generate questions and corresponding answers based on the content of each chunk.

---

### Step 3: Embedding and ElasticSearch
- **Objective**: Enable efficient retrieval of relevant chunks.  
- **Implementation**:
  - Pre-embed chunks into a vector space using a transformer-based embedding model.
  - Use **ElasticSearch** to index these embeddings and perform similarity-based retrieval.

---

### Step 4: Retrieval
- **Objective**: Retrieve the top-k chunks most relevant to a given question.  
- **Implementation**:
  - Perform vector similarity search using ElasticSearch.
  - Limit retrieval to `k` chunks (default: `k=3`).

---

### Step 5: RAG Answer Generation
- **Objective**: Generate a final answer using the retrieved chunks.  
- **Implementation**:
  - Pass the retrieved chunks and the input question to the LLM.
  - Generate the final answer using the RAG paradigm.

---

### Step 6: Evaluation
- **Objective**: Assess the performance of the RAG pipeline using quantitative and qualitative metrics.  
- **Metrics**:
  - **Recall**: Measures how many of the relevant chunks were retrieved.
  - **Precision**: Measures the proportion of retrieved chunks that are relevant (evaluated at specific `k` values).
  - **F1-Score**: A harmonic mean of Recall and Precision.
  - **Answer Relevance**: Assesses whether the generated answer addresses the question.
  - **Faithfulness**: Evaluates if the answer is consistent with the retrieved chunks.

---

## Testing Process

1. **Single-Chunk Testing**  
   - Each question has one corresponding Original Chunk.
   - Evaluate Recall and Precision (focus on Recall as the primary metric).  
   - Precision is tested specifically when `k=1`.

2. **Multi-Chunk Testing**  
   - For questions requiring multiple chunks as evidence:
     - Limit reference chunks to **3 per question**.
     - Limit ElasticSearch retrieval to **k=3**.
     - Evaluate Precision and Recall, and compute **F1-Score**.

3. **LLM as a Judge**  
   - Use the LLM to compare the **generated answer** with the **original answer** from the dataset.  
   - The LLM determines whether the two answers are equivalent.

---

## Limitations

1. **Extreme Recall Values Due to Single Original Chunk**  
   - The current method assumes that there is only **one Original Chunk** as the correct answer for each question.  
   - As a result, if the Original Chunk is found in the retrieved results, **Recall = 1**, and if it is not found, **Recall = 0**.  
   - This makes it difficult to perform a granular evaluation of actual RAG performance and fails to reflect situations where multiple pieces of evidence (i.e., multiple chunks) are needed to answer a question.

2. **Precision Degradation Due to kNN (Search Range) Settings**  
   - In ElasticSearch, increasing **k** (the number of retrieved chunks) results in retrieving more chunks. While this can improve **Recall**, it also increases the number of irrelevant chunks, leading to a decrease in **Precision**.  
   - This issue is particularly severe when there is only one Original Chunk per question because, as more chunks are retrieved, all except the single correct chunk are considered irrelevant, causing **Precision to drop sharply**.

3. **Insufficient Reflection of Multi-Chunk Query Scenarios**  
   - In real-world tasks or services, it is common for a single question to require evidence from **multiple chunks**.  
   - If only a single Original Chunk is considered, it becomes difficult to properly handle and evaluate **complex queries** and to provide actionable feedback for improving model performance.

4. **Need for Dataset Construction**  
   - To address the above limitations, it is necessary to construct a separate dataset that sets **multiple Original Chunks per question**.  
   - This could involve either (1) **manually collecting additional data** or (2) designing prompts that include multiple pieces of evidence per question to ensure that the dataset contains cases where answers are constructed from multiple chunks.

---

### Verification Approach at the Company

1. **Focusing on Recall for Retrieval**  
   - During the evaluation of retrieval performance, the primary metric selected was **Recall**.  
   - The reasoning behind this choice is that as the size of the LLM’s parameters increases, if the retrieved document contains the relevant information, the model tends to provide a good answer. Consequently, **Recall** was deemed more critical than Precision for this particular test.  
   - **Precision** was considered **only when k=1**, meaning that in cases where only one chunk is retrieved, the team did examine how precise that single retrieval was.

2. **Using LLM as a Judge**  
   - To conduct a more intuitive test, the **final generated answer** was compared against the **original Q&A answer** from the initial dataset.  
   - This comparison was done using an **“LLM as a Judge”** approach, where the model itself determined whether the two answers were essentially the same or not, serving as an automated grading system.

3. **Handling Multi-Chunk Answers**  
   - For questions that required multiple chunks to form a complete answer, the author manually created **100 test problems**.  
   - Each question was designed so that it would reference **exactly three chunks**, and the **Vector Search** was also limited to retrieving **three chunks**.  
   - In this setup, **Precision** was measured at **k=3**, and an **F1-Score** was computed to evaluate the balance between Recall and Precision under these multi-chunk constraints.

By incorporating these steps—focusing on Recall for the bulk of retrieval testing, using an LLM as a judge for direct answer verification, and constructing a targeted multi-chunk dataset for more nuanced performance evaluation—the author ensured a comprehensive assessment of the system’s capabilities.