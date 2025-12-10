import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

DISPLAY_TOP = 25          # we want 25 grouped rows in the end
HASH_DIM = 4096           # keeps RAM lower


def parse_course_level(code: str | int | float):
    """Get 100, 200, 300, ... from the first digit of the course number."""
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

    # figure out course column name
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
    X = normalize(X, axis=1)  # each row length 1

    # group row indices by course
    course_to_rows = defaultdict(list)
    for i, c in df[course_col].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())

    # precompute level → courses mapping once
    level_to_courses = defaultdict(list)
    for c in courses:
        lvl = parse_course_level(c)
        if lvl is not None:
            level_to_courses[lvl].append(c)

    # ---------- COURSE vs COURSE (guarantee up to 25 groups) ----------
    course_rows = []
    print("Computing course–course similarities with grouping up to 25...")

    for course_a in courses:
        lvl_a = parse_course_level(course_a)
        idx_a = course_to_rows[course_a]
        if not idx_a or lvl_a is None:
            continue

        # mean vector for base course
        mean_a = X[idx_a].mean(axis=0)
        mean_a_vec = np.asarray(mean_a).ravel()

        # candidates: all same-level courses except itself
        same_level_courses = [
            c for c in level_to_courses[lvl_a] if c != course_a
        ]
        if not same_level_courses:
            continue

        sims = []
        for course_b in same_level_courses:
            idx_b = course_to_rows[course_b]
            if not idx_b:
                continue
            mean_b = X[idx_b].mean(axis=0)
            mean_b_vec = np.asarray(mean_b).ravel()
            sim = float(mean_a_vec.dot(mean_b_vec))
            sims.append((course_b, sim))

        if not sims:
            continue

        # sort by similarity descending
        sims.sort(key=lambda x: x[1], reverse=True)

        grouped_keys = []     # order of unique rounded sims
        grouped_key_set = set()

        for course_b, sim in sims:
            key = round(sim, 3)  # group rule A: similarity rounded to 3 decimals

            if key not in grouped_key_set:
                # if this is the 26th new group, stop (we only want first 25 groups)
                if len(grouped_keys) >= DISPLAY_TOP:
                    break
                grouped_keys.append(key)
                grouped_key_set.add(key)

            # now key is in the first <=25 groups, add this row
            course_rows.append(
                {"base_course": course_a, "other_course": course_b, "overall": sim}
            )

    course_sim_df = pd.DataFrame(course_rows)
    course_sim_df.to_csv("course_similarity_top.csv", index=False)
    print(f"Saved course_similarity_top.csv ({len(course_sim_df)} rows)")

    # ---------- CLO vs CLO (guarantee up to 25 CLO text groups) ----------
    print("Computing CLO–CLO similarities with grouping up to 25 texts...")
    clo_rows = []

    for base_idx in range(n):
        if base_idx % 1000 == 0:
            print(f"  CLO {base_idx}/{n}")

        base_course = df.at[base_idx, course_col]
        lvl = parse_course_level(base_course)
        if lvl is None:
            continue

        # candidates: all CLOs in same-level courses EXCLUDING same course
        same_level_courses = [
            c for c in level_to_courses[lvl] if c != base_course
        ]
        if not same_level_courses:
            continue

        cand_indices = []
        for c in same_level_courses:
            cand_indices.extend(course_to_rows[c])

        if not cand_indices:
            continue

        base_vec = X[base_idx]          # 1 x D
        cand_mat = X[cand_indices]      # M x D

        sims_sparse = base_vec.dot(cand_mat.T)
        sims = np.asarray(sims_sparse.todense()).ravel()

        pairs = list(zip(cand_indices, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)

        seen_texts = set()
        unique_text_count = 0

        for other_idx, sim in pairs:
            clo_text = df.at[other_idx, "CLO_TEXT"]

            # group rule 2A: by exact CLO text
            if clo_text not in seen_texts:
                # if this would be the 26th unique text, stop
                if unique_text_count >= DISPLAY_TOP:
                    break
                seen_texts.add(clo_text)
                unique_text_count += 1

            # store this row (may be another course with same text)
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
