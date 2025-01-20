## Overview

In an era of increasingly capable language models, systematic evaluation is critical to understanding their strengths, weaknesses, and applicability in real-world scenarios. This validation framework provides tailored services for evaluating language models across diverse benchmarks. Each service is designed to handle a specific dataset, preprocess the data for model compatibility, and assess the model's performance by comparing predictions against ground truth labels.


### Purpose of Each Dataset

#### **1. ARC (Allen AI Reasoning Challenge)**
- **Objective**: To evaluate the scientific reasoning capabilities of models.
- **Description**: This dataset contains multiple-choice science questions aimed at assessing a model's ability to reason about scientific facts and principles.
- **Typical Use Case**: Testing models in academic or knowledge-intensive scenarios where understanding scientific concepts is critical.

---

#### **2. GSM8K (Grade School Math 8K)**
- **Objective**: To measure a model's ability to solve mathematical problems requiring multi-step reasoning.
- **Description**: This dataset consists of grade-school-level math problems, paired with detailed explanations and numeric solutions.
- **Typical Use Case**: Evaluating a model's ability to handle quantitative reasoning and arithmetic.

---

#### **3. HellaSwag**
- **Objective**: To test contextual reasoning and commonsense understanding.
- **Description**: This dataset presents incomplete sentences followed by multiple possible continuations, requiring the model to choose the most appropriate one.
- **Typical Use Case**: Assessing a model's ability to predict contextually appropriate outcomes based on prior context.

---

#### **4. MMLU (Massive Multitask Language Understanding)**
- **Objective**: To benchmark multitask language understanding across a wide range of subjects and difficulty levels.
- **Description**: The dataset includes multiple-choice questions from diverse domains, such as history, science, and engineering.
- **Typical Use Case**: Testing the general knowledge and reasoning capabilities of language models across varied disciplines.

---

#### **5. TruthfulMCQA**
- **Objective**: To evaluate a model's ability to produce truthful and factually accurate responses.
- **Description**: This dataset features multiple-choice questions designed to challenge the model to avoid common misconceptions or falsehoods.
- **Typical Use Case**: Validating models for use in domains where factual accuracy is critical, such as education or healthcare.

---

#### **6. Winogrande**
- **Objective**: To assess a model's ability to resolve pronoun ambiguities in sentences.
- **Description**: The dataset includes sentences with ambiguous pronouns or blanks, accompanied by two possible completions. The task is to select the most contextually appropriate completion.
- **Typical Use Case**: Evaluating the model's contextual understanding and ability to resolve linguistic ambiguities.


## ARC Validation Service

The **ARCValidService** evaluates models on the ARC (Allen AI Reasoning Challenge) dataset, designed for scientific reasoning problems.

### Dataset Structure
- **`question`**: A science-related question.
- **`choices`**: A dictionary with:
  - **`label`**: Letter labels for each choice (e.g., "A", "B", "C").
  - **`text`**: The corresponding choice text.
- **`answerKey`**: The correct answer label.

#### Example
```json
{
  "question": "What is the boiling point of water at sea level?",
  "choices": {
    "label": ["A", "B", "C"],
    "text": ["100°C", "0°C", "50°C"]
  },
  "answerKey": "A"
}
```

### Preprocessing
- Combines the question and choices into a `user_prompt`.
- Labels are dynamically generated as `["A", "B", "C"]`.
- Extracts the correct answer from `answerKey`.

---

## GSM8K Validation Service

The **GSM8KValidService** evaluates mathematical reasoning using grade-school-level math problems.

### Dataset Structure
- **`question`**: The math problem.
- **`answer`**: An explanation followed by the solution in the format `\n#### {solution}`.

#### Example
```json
{
  "question": "At 30, Anika is 4/3 the age of Maddie. What would be their average age in 15 years?",
  "answer": "How old will Anika be in 15 years?...#### 50"
}
```

### Preprocessing
- Extracts the question as the `user_prompt`.
- Parses the numeric solution from the `answer` using a regex.

---

## HellaSwag Validation Service

The **HellaSwagValidService** evaluates contextual reasoning by predicting the most likely continuation of a sentence.

### Dataset Structure
- **`ctx_a`**: Initial context.
- **`ctx_b`**: Continuation fragment.
- **`ctx`**: Combined context.
- **`endings`**: A list of possible continuations.
- **`label`**: The index of the correct continuation.

#### Example
```json
{
  "ctx_a": "A man is sitting on a roof.",
  "ctx_b": "he",
  "ctx": "A man is sitting on a roof. he",
  "endings": ["is using wrap...", "is ripping tiles...", "is holding...", "starts pulling roofing..."],
  "label": "3"
}
```

### Preprocessing
- Combines `ctx` and `endings` into a `user_prompt`.
- Labels are dynamically assigned as `["A", "B", "C", "D"]`.
- Converts the `label` index to the corresponding letter.

---

## MMLU Validation Service

The **MMLUValidService** evaluates multitask language understanding across diverse domains.

### Dataset Structure
- **`question`**: The question text.
- **`options`**: A list of possible answers.
- **`answer`**: The correct answer as a letter.

#### Example
```json
{
  "question": "What is the capital of France?",
  "options": ["Berlin", "Madrid", "Paris", "Rome"],
  "answer": "C"
}
```

### Preprocessing
- Combines the `question` and `options` into a `user_prompt`.
- Dynamically generates labels as `["A", "B", "C", "D"]`.
- Extracts the correct answer from `answer`.

---

## TruthfulMCQA Validation Service

The **TruthfulMCQAValidService** evaluates the factual correctness of answers to multiple-choice questions.

### Dataset Structure
- **`question`**: The question text.
- **`choices`**: A list of possible answers.
- **`label`**: The index of the correct answer.

#### Example
```json
{
  "question": "What is the smallest country in the world that is at least one square mile in area?",
  "choices": [
    "Monaco",
    "Vatican City",
    "United States",
    "Nauru"
  ],
  "label": "1"
}
```

### Preprocessing
- Combines the `question` and `choices` into a `user_prompt`.
- Dynamically assigns labels (`["A", "B", "C", "D"]`).
- Maps the numeric `label` to the corresponding letter (`real_answer`).

---

## Winogrande Validation Service

The **WinograndeValidService** evaluates models on resolving ambiguities in pronoun references.

### Dataset Structure
- **`sentence`**: A sentence with an ambiguous pronoun or blank.
- **`option1`**: The first possible completion.
- **`option2`**: The second possible completion.
- **`answer`**: The correct answer index (`1` or `2`).

#### Example
```json
{
  "sentence": "John moved the couch from the garage to the backyard. The _ is small.",
  "option1": "garage",
  "option2": "backyard",
  "answer": "1"
}
```

### Preprocessing
- Combines `sentence`, `option1`, and `option2` into a `user_prompt`.
- Assigns labels as `["1", "2"]`.
- Extracts the correct answer from `answer`.

---

## Evaluation Process (Common to All Services)

1. **Input Preparation**:
   - Each example is preprocessed to create a `system_prompt` and `user_prompt`.
   - Labels and the ground truth (`real_answer`) are extracted.

2. **Model Prediction**:
   - The model processes the `system_prompt` and `user_prompt` to generate logits.
   - Top predictions are decoded and matched with the labels.

3. **Scoring**:
   - The predicted answer is compared to the `real_answer`.
   - Accuracy is calculated as the ratio of correct predictions to the total number of examples.

#### Example Output
```plaintext
0. Output: A, Reference: A :: O
1. Output: B, Reference: C :: X
Score: 0.5
```