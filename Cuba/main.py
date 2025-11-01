with open("Cuba/input.txt", "r") as f:
    text = f.read()

cleaned_text = text.replace('.', '')

with open("Cuba/output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)
