import nltk
from nltk.corpus import stopwords

# Download the stopwords list from NLTK
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

with open("input.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    words = line.strip().split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_lines.append(" ".join(filtered_words))

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))
