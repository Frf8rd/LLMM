# Text Cleaning Task – “Stop Signs”

## Task Overview
The goal of this task was to process a block of text from `input.txt` and produce a cleaned version according to the hint provided in the image `hint.jpg`. The hint focused on "stop signs," which indicated **removing periods (dots) from the text**.

## Understanding the Task
- **Input:** `participant_input/input.txt` – a plain text file containing multiple sentences.
- **Hint:** `participant_input/hint.jpg` – suggested that "stop signs" (periods) should be removed.
- **Output:** `output.txt` – cleaned text with periods removed, original line breaks preserved, and words separated by single spaces.

## How the task was solved
1. Opened the input file using Python's `with open(...)` in UTF-8 encoding.
2. Read the entire content of the file.
3. Removed all periods (`.`) from the text using `str.replace(".", "")`.
4. Wrote the cleaned text to `output.txt`, preserving the original line breaks.
