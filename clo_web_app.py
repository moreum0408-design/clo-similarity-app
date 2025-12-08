import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

TOP_COURSE = 20   # top courses per course
TOP_CLO = 10      # top CLOs per CLO (kept small so CSV < 25MB)
HASH_DIM = 4096   # keeps RAM low


def parse_course_level(code):
    if not isinstance(code, str):
        code = str(code)
    m = re.search(r"(\d+)", code)
    if not m:
        return None
    return int(m.group(1)[0]) * 100


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    print("Reading Excel...")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    if "COURSE" in df.columns:
        course_col = "COURSE"
    elif "Course" in df.columns:
        course_col = "Course"
    else:
        raise ValueError("No COURSE or Course column in Excel")

    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].astype(str)

    n = len(df)
    print(f"{n} CLO rows.")

    # ---------- TF-IDF-ish with hashing (small memory) ----------
    print("Vectorizing text with HashingVectorizer...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(df["CLO_TEXT"])
    X = normalize(X, axis=1)  # each row len 1

    # group indices by course
    course_to_rows = defaultdict(list)
    for i, c in df[course_col].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())

    # ---------- COURSE vs COURSE ----------
    course_rows = []
    print("Computing course–course similarities...")
    for course_a in courses:
        lvl_a = parse_course_level(course_a)
        idx_a = course_to_rows[course_a]
        if not idx_a or lvl_a is None:
            continue

        mean_a = X[idx_a].mean(axis=0)          # 1 x D sparse
        mean_a_vec = np.asarray(mean_a).ravel() # dense 1D

        sims = []
        for course_b in courses:
            if course_b == course_a:
                continue
            if parse_course_level(course_b) != lvl_a:
                continue
            idx_b = course_to_rows[course_b]
            if not idx_b:
                continue
            mean_b = X[idx_b].mean(axis=0)
            mean_b_vec = np.asarray(mean_b).ravel()
            sim = float(mean_a_vec @ mean_b_vec)
            sims.append((course_b, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        for course_b, sim in sims[:TOP_COURSE]:
            course_rows.append(
                {"base_course": course_a, "other_course": course_b, "overall": sim}
            )

    course_sim_df = pd.DataFrame(course_rows)
    course_sim_df.to_csv("course_similarity_top.csv", index=False)
    print(f"Saved course_similarity_top.csv ({len(course_sim_df)} rows)")

    # ---------- CLO vs CLO ----------
    print("Computing CLO–CLO similarities (top 10) ...")
    clo_rows = []

    # quick lookup of top same-level courses
    for base_idx in range(n):
        if base_idx % 1000 == 0:
            print(f"  CLO {base_idx}/{n}")

        base_course = df.at[base_idx, course_col]
        lvl = parse_course_level(base_course)
        if lvl is None:
            continue

        subset = course_sim_df[course_sim_df["base_course"] == base_course]
        if subset.empty:
            continue

        top_courses = (
            subset.sort_values("overall", ascending=False)["other_course"]
            .head(TOP_COURSE)
            .tolist()
        )
        allowed_courses = set(top_courses)
        if not allowed_courses:
            continue

        # all candidate CLO indices from those courses
        cand_indices = [
            i for i, c in df[course_col].items() if c in allowed_courses
        ]
        if not cand_indices:
            continue

        base_vec = X[base_idx]                # 1 x D sparse, normalized
        cand_mat = X[cand_indices]            # M x D sparse, normalized

        sims_sparse = base_vec.dot(cand_mat.T)
        sims = np.asarray(sims_sparse.todense()).ravel()  # length M

        pairs = list(zip(cand_indices, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)
        for other_idx, sim in pairs[:TOP_CLO]:
            clo_rows.append(
                {
                    "base_idx": base_idx,
                    "other_idx": int(other_idx),
                    "similarity": float(sim),
                }
            )

    clo_sim_df = pd.DataFrame(clo_rows)
    clo_sim_df.to_csv("clo_similarity_top.csv", index=False)
    print(f"Saved clo_similarity_top.csv ({len(clo_sim_df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
