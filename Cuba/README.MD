# Text Cleaning Task: Stopwords Removal

This document outlines the objective, input, approach, and output for the text cleaning task focused on **stopwords removal**. The goal is to preprocess a text file by eliminating common, non-essential English words to reduce noise and emphasize key content.

---

##  Objective

To clean the provided plain text file by removing **stopwords** (common English words like `the`, `is`, `a`, `and`, etc.) while ensuring the **meaning and structural integrity** of the original content are preserved.

---

##  Input and Output

### Input
* **File:** `participant_input/input.txt`
* **Format:** Plain text
* **Content:** Multiple sentences detailing the **Sigmoid NGO** and the **LMML competition**.

### Output
* **File:** `output.txt`
* **Format:** Plain text
* **Encoding:** **UTF-8**
* **Requirements:**
    * All identified stopwords are removed.
    * **Original line breaks** are precisely preserved.
    * Remaining words are separated by **single spaces**.

---

## Approach

### 1. Stopwords Identification
A comprehensive list of common English stopwords is used (e.g., `the`, `is`, `in`, `of`, `and`, `are`, etc.). These words are deemed to not contribute **significant meaning** to the text.

### 2. Line-by-Line Processing
To meet the requirement of preserving original structure, the input file is read and processed **line by line**. Each line is treated as an independent unit.

### 3. Tokenization and Filtering
* Each line is tokenized into individual words.
* Every word is checked against the stopwords list.
* Only words that **are NOT** on the stopwords list are retained. This step ensures that **meaningful content** is highlighted and kept.

### 4. Structure Preservation and Formatting
* After filtering, the remaining words on each line are joined back together using a **single space** as a separator.
* The original line breaks are preserved by writing the processed line immediately to the output file, maintaining the original paragraph and sentence structure.

---
