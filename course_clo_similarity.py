# clo_similarity.py
import sys
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === SETTINGS (change only if your file/columns differ) ===
EXCEL_PATH = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"          # this is the sheet name in your file
COL_PROGRAM = "CRSE"
COL_TEXT = "CLO_TEXT"
# ==========================================================

# Load & prep
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, usecols=[COL_PROGRAM, COL_TEXT])
df = df.dropna(subset=[COL_TEXT])
df[COL_TEXT] = df[COL_TEXT].astype(str).str.strip()

# Concatenate all CLO text per program
programs = df.groupby(COL_PROGRAM)[COL_TEXT].apply(lambda s: " ".join(s)).reset_index()
programs.index = programs[COL_PROGRAM]

# Vectorize once (char n-grams = no keywords, just phrasing similarity)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
X = vectorizer.fit_transform(programs[COL_TEXT])

def suggest(name, n=8):
    return get_close_matches(name, programs.index.tolist(), n=n, cutoff=0.5)

def get_similarity(p1, p2):
    try:
        i1 = programs.index.get_loc(p1)
    except KeyError:
        cand = suggest(p1)
        raise KeyError(f"Course not found: {p1}\nDid you mean:\n  - " + "\n  - ".join(cand) if cand else "  (no close matches)")
    try:
        i2 = programs.index.get_loc(p2)
    except KeyError:
        cand = suggest(p2)
        raise KeyError(f"Course not found: {p2}\nDid you mean:\n  - " + "\n  - ".join(cand) if cand else "  (no close matches)")
    score = float(cosine_similarity(X[i1], X[i2])[0][0])
    return score

def main():
    if len(sys.argv) == 3:
        p1, p2 = sys.argv[1], sys.argv[2]
        score = get_similarity(p1, p2)
        print(f"\nSimilarity between\n  '{p1}'\n  '{p2}'\n= {score:.3f}\n")
        return

    print("\nCLO course similarity — type two course codes. Case and format must be identical as spreadsheet (exact or close).")
    print("Type q to quit.\n")
    while True:
        p1 = input("First course: ").strip()
        if p1.lower() == "q": break
        p2 = input("Second course: ").strip()
        if p2.lower() == "q": break
        try:
            score = get_similarity(p1, p2)
            print(f"\nSimilarity = {score:.3f}\n")
        except KeyError as e:
            print("\n" + str(e) + "\n")

if __name__ == "__main__":
    main()
