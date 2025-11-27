# clo_similarity.py — program/pair similarity + scanlevel (course + CLO-level), with school weights

import sys, argparse, re
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== SETTINGS (change if your sheet/columns differ) =======================
EXCEL_PATH = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

COL_SCHOOL  = "COLLEGE"   # college/school column
COL_PROGRAM = "Program"
COL_COURSE  = "Course"
COL_TEXT    = "CLO_TEXT"
# ===========================================================================

# ===== SCHOOL WEIGHTS ======================================================
SCHOOL_WEIGHTS = {
    "Helms School of Government": 15,
    "School of Business": 14,
    "School of Law": 13,
    "School of Behavioral Sciences": 12,
    "College of Arts & Sciences": 11,
    "School of Education": 10,
    "School of Communications & the Arts": 9,
    "School of Divinity": 8,
    "School of Engineering": 7,
    "School of Health Sciences": 6,
    "College of Applied Studies & Academic Success": 5,
    "School of Music": 4,
    "School of Aeronautics": 3,
    "School of Nursing": 2,
    "College of Osteopathic Medicine": 1,
}
MAX_SCHOOL_WEIGHT = max(SCHOOL_WEIGHTS.values())


def get_school_weight(name: str) -> float:
    if not isinstance(name, str):
        return 1.0
    name = name.strip()
    return float(SCHOOL_WEIGHTS.get(name, 1.0))


# ===== TEXT HELPERS ========================================================
DASHES = "\u2013\u2014\u2212\u2012\u2010\u2011"  # – — − ‒ ‐ -


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(f"[{DASHES}]", "-", s)   # unify fancy dashes to '-'
    s = re.sub(r"\s+", " ", s)          # collapse spaces
    return s


def canon(s: str) -> str:
    return normalize_text(s).casefold()


def parse_pair_input(raw: str) -> str:
    """
    Accept 'Prog | CRSE' or 'Prog, CRSE' or 'Prog::CRSE' or 'Prog / CRSE'
    and return 'Prog | CRSE'.
    (This is only for COURSE-level keys, not CLO-level.)
    """
    raw = normalize_text(raw)
    parts = re.split(r"\s*(\||,|::|/)\s*", raw)
    tokens = [t for t in parts if t not in {"|", ",", "::", "/"}]
    if len(tokens) < 2:
        return raw
    prog, crse = tokens[0], tokens[-1]
    return f"{prog} | {crse}"


# ===== COURSE-LEVEL UTILITIES =============================================
NUM_RE = re.compile(r"(\d{3,})")  # capture 3+ digits (e.g., 705, 210, 1010)


def course_level_digit(course: str):
    """Return the first digit of the numeric part (e.g., AVIA705 -> '7'; ACCT210 -> '2'). None if not found."""
    course = normalize_text(str(course))
    m = NUM_RE.search(course)
    if not m:
        return None
    return m.group(1)[0]


def course_prefix(course: str):
    """Alphabetic prefix (e.g., AVIA705 -> 'AVIA')."""
    m = re.match(r"([A-Za-z]+)", str(course))
    return m.group(1) if m else ""


# ===== LOADING / CORPORA ===================================================
def load_data(excel_source=None):
    src = excel_source or EXCEL_PATH
    use_cols = [COL_SCHOOL, COL_PROGRAM, COL_COURSE, COL_TEXT]

    df = pd.read_excel(src, sheet_name=SHEET_NAME, usecols=use_cols)
    df = df.dropna(subset=[COL_TEXT])

    df[COL_SCHOOL]  = df[COL_SCHOOL].astype(str).map(normalize_text)
    df[COL_PROGRAM] = df[COL_PROGRAM].astype(str).map(normalize_text)
    df[COL_COURSE]  = df[COL_COURSE].astype(str).map(normalize_text)
    df[COL_TEXT]    = df[COL_TEXT].astype(str).map(str.strip)

    return df


def build_program_corpus(df):
    progs = df.groupby(COL_PROGRAM)[COL_TEXT].apply(lambda s: " ".join(s)).reset_index()
    progs["KEY"] = progs[COL_PROGRAM]
    progs["CANON"] = progs["KEY"].map(canon)
    progs = progs.set_index("CANON")
    return progs


def build_pair_corpus(df):
    """
    Course-level corpus: all CLOs for a (School, Program, Course) merged.
    """
    pairs = (
        df.groupby([COL_SCHOOL, COL_PROGRAM, COL_COURSE])[COL_TEXT]
          .apply(lambda s: " ".join(s))
          .reset_index()
    )
    pairs["KEY"] = pairs[COL_PROGRAM] + " | " + pairs[COL_COURSE]
    pairs["CANON"] = pairs["KEY"].map(canon)
    pairs["LEVEL"] = pairs[COL_COURSE].map(course_level_digit)
    pairs["PREFIX"] = pairs[COL_COURSE].map(course_prefix)
    pairs["SCHOOL_WEIGHT"] = pairs[COL_SCHOOL].map(get_school_weight)
    pairs = pairs.set_index("CANON")
    return pairs


def build_clo_corpus(df):
    """
    CLO-level corpus.
    Each row = ONE CLO_TEXT with its Program, Course, College, CLO index.
    KEY looks like: 'Doctor of Criminal Justice (D.C.J.) | CJUS887 | CLO1'
    """
    df = df.copy()
    df["CLO_INDEX"] = df.groupby([COL_PROGRAM, COL_COURSE]).cumcount() + 1

    df["KEY"] = (
        df[COL_PROGRAM] + " | " + df[COL_COURSE] +
        " | CLO" + df["CLO_INDEX"].astype(str)
    )
    df["CANON"] = df["KEY"].map(canon)
    df["LEVEL"] = df[COL_COURSE].map(course_level_digit)
    df["PREFIX"] = df[COL_COURSE].map(course_prefix)
    df["SCHOOL_WEIGHT"] = df[COL_SCHOOL].map(get_school_weight)

    clo = df.set_index("CANON")
    return clo


def vectorize(text_series):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X = vec.fit_transform(text_series)
    return vec, X


def suggest(raw_key, table, n=10):
    cands = get_close_matches(canon(raw_key), list(table.index), n=n, cutoff=0.5)
    return [table.loc[c, "KEY"] for c in cands] if cands else []


def pos_map_from_index(index):
    return {k: i for i, k in enumerate(index)}


def overall_stats(df, top=None, col="WSimilarity"):
    """Compute stats on df[col]. If df empty, returns zeros."""
    if df is None or df.empty or col not in df.columns:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "topk_mean": 0.0, "n": 0}
    sims = df[col].to_numpy()
    sims_sorted = np.sort(sims)[::-1]
    topk_mean = float(np.mean(sims_sorted[:top])) if (top and top > 0) else float(np.mean(sims_sorted))
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "max": float(np.max(sims)),
        "topk_mean": topk_mean,
        "n": int(len(sims)),
    }


# ===== CORE LOOKUPS ========================================================
def get_score(a_raw, b_raw, table, X):
    a_key = parse_pair_input(normalize_text(a_raw))
    b_key = parse_pair_input(normalize_text(b_raw))
    a_idx = canon(a_key)
    b_idx = canon(b_key)
    if a_idx not in table.index:
        raise KeyError(
            "Not found: " + a_raw +
            ("\nDid you mean:\n  - " + "\n  - ".join(suggest(a_raw, table)) if suggest(a_raw, table) else "")
        )
    if b_idx not in table.index:
        raise KeyError(
            "Not found: " + b_raw +
            ("\nDid you mean:\n  - " + "\n  - ".join(suggest(b_raw, table)) if suggest(b_raw, table) else "")
        )
    pmap = pos_map_from_index(table.index)
    i1, i2 = pmap[a_idx], pmap[b_idx]
    return float(cosine_similarity(X[i1], X[i2])[0][0])


def scan_same_level_course(anchor_raw, pairs, X, same_prefix=False, min_score=0.0, alpha=1.0):
    """
    COURSE-LEVEL:
    Compare anchor (Program | Course) to ALL same-level courses, weighted by school.
    """
    use_weights = True

    key = parse_pair_input(normalize_text(anchor_raw))
    idx = canon(key)
    if idx not in pairs.index:
        raise KeyError(
            "Not found: " + anchor_raw +
            ("\nDid you mean:\n  - " + "\n  - ".join(suggest(anchor_raw, pairs)) if suggest(anchor_raw, pairs) else "")
        )

    anchor_row = pairs.loc[idx]
    level = anchor_row["LEVEL"]
    prefix = anchor_row["PREFIX"]
    if level is None:
        raise ValueError(f"Could not detect level digit for course in '{anchor_row['KEY']}'.")

    candidates = pairs[pairs["LEVEL"] == level].copy()
    if same_prefix:
        candidates = candidates[candidates["PREFIX"] == prefix]

    candidates = candidates[candidates["KEY"] != anchor_row["KEY"]]
    if candidates.empty:
        return pd.DataFrame(columns=["KEY", "Similarity", "WSimilarity", "LEVEL", "PREFIX",
                                     COL_SCHOOL, "SCHOOL_WEIGHT", COL_TEXT])

    pmap = pos_map_from_index(pairs.index)
    anchor_vec = X[pmap[idx]]
    cand_rows = [pmap[i] for i in candidates.index]
    sims = cosine_similarity(anchor_vec, X[cand_rows]).ravel()

    weights = candidates["SCHOOL_WEIGHT"].astype(float).to_numpy()
    if use_weights:
        factors = (weights / MAX_SCHOOL_WEIGHT) ** float(alpha)
        w_sims = sims * factors
    else:
        w_sims = sims

    out = candidates.copy()
    out = out.assign(Similarity=sims, WSimilarity=w_sims)
    out = out[out["WSimilarity"] >= min_score].sort_values("WSimilarity", ascending=False)
    return out[["KEY", "Similarity", "WSimilarity", "LEVEL", "PREFIX", COL_SCHOOL, "SCHOOL_WEIGHT", COL_TEXT]]


def scan_same_level_clo(anchor_raw, clo, X, same_prefix=False, min_score=0.0, alpha=1.0):
    """
    CLO-LEVEL:
    Compare ONE specific CLO (Program | Course | CLO#) against
    ALL other CLOs with the SAME course level digit.
    """
    use_weights = True

    key = normalize_text(anchor_raw)
    idx = canon(key)
    if idx not in clo.index:
        raise KeyError(
            "Not found: " + anchor_raw +
            ("\nDid you mean:\n  - " + "\n  - ".join(suggest(anchor_raw, clo)) if suggest(anchor_raw, clo) else "")
        )

    anchor_row = clo.loc[idx]
    level = anchor_row["LEVEL"]
    prefix = anchor_row["PREFIX"]
    if level is None:
        raise ValueError(f"Could not detect level digit for course in '{anchor_row['KEY']}'.")

    candidates = clo[clo["LEVEL"] == level].copy()
    if same_prefix:
        candidates = candidates[candidates["PREFIX"] == prefix]

    candidates = candidates[candidates["KEY"] != anchor_row["KEY"]]
    if candidates.empty:
        return pd.DataFrame(columns=[
            "KEY", "Similarity", "WSimilarity", "LEVEL", "PREFIX",
            COL_SCHOOL, "SCHOOL_WEIGHT", COL_TEXT
        ])

    pmap = pos_map_from_index(clo.index)
    anchor_vec = X[pmap[idx]]
    cand_rows = [pmap[i] for i in candidates.index]
    sims = cosine_similarity(anchor_vec, X[cand_rows]).ravel()

    weights = candidates["SCHOOL_WEIGHT"].astype(float).to_numpy()
    if use_weights:
        factors = (weights / MAX_SCHOOL_WEIGHT) ** float(alpha)
        w_sims = sims * factors
    else:
        w_sims = sims

    out = candidates.copy()
    out = out.assign(Similarity=sims, WSimilarity=w_sims)
    out = out[out["WSimilarity"] >= min_score].sort_values("WSimilarity", ascending=False)
    return out[[
        "KEY", "Similarity", "WSimilarity", "LEVEL", "PREFIX",
        COL_SCHOOL, "SCHOOL_WEIGHT", COL_TEXT
    ]]


# ===== INTERACTIVE MODES ===================================================
def run_program_mode():
    df = load_data()
    progs = build_program_corpus(df)
    _, X = vectorize(progs[COL_TEXT])
    print("\nType q to quit. Type '?find keywords' to search program names.\n")
    print("Compare two PROGRAMS by CLO language.\n")
    while True:
        a = input("First program: ").strip()
        if a.lower() == "q":
            break
        if a.startswith("?find"):
            q = a[5:].strip().casefold()
            hits = [progs.loc[i, "KEY"] for i in progs.index if q in i][:20]
            if hits:
                print("\nMatches:\n  - " + "\n  - ".join(hits) + "\n")
            else:
                print("\n(no matches)\n")
            continue
        b = input("Second program: ").strip()
        if b.lower() == "q":
            break
        try:
            s = get_score(a, b, progs, X)
            print(f"\nSimilarity = {s:.3f}\n")
        except KeyError as e:
            print("\n" + str(e) + "\n")


def run_pair_mode():
    df = load_data()
    pairs = build_pair_corpus(df)
    _, X = vectorize(pairs[COL_TEXT])
    print("\nType q to quit. Type '?find ACCT' or '?find 705' to search.\n")
    print("Compare two (PROGRAM | COURSE) pairs.\n")
    while True:
        a = input("First pair (Program | Course): ").strip()
        if a.lower() == "q":
            break
        if a.startswith("?find"):
            q = canon(a[5:].strip())
            hits = [pairs.loc[i, "KEY"] for i in pairs.index if q in i][:20]
            if hits:
                print("\nMatches:\n  - " + "\n  - ".join(hits) + "\n")
            else:
                print("\n(no matches)\n")
            continue
        b = input("Second pair (Program | Course): ").strip()
        if b.lower() == "q":
            break
        try:
            s = get_score(a, b, pairs, X)
            print(f"\nSimilarity = {s:.3f}\n")
        except KeyError as e:
            print("\n" + str(e) + "\n")


def run_scanlevel_mode(same_prefix=False, top=50, min_score=0.0,
                       save_csv=None, overall_only=False, topk_for_overall=None,
                       alpha=1.0):
    """
    Single mode that accepts EITHER:
      - 'Program | COURSE'          -> course-level similarity (all CLOs)
      - 'Program | COURSE | CLO#'   -> specific CLO similarity
    """
    df = load_data()
    pairs = build_pair_corpus(df)   # course-level
    clo   = build_clo_corpus(df)    # clo-level
    _, X_pairs = vectorize(pairs[COL_TEXT])
    _, X_clo   = vectorize(clo[COL_TEXT])

    print("\nType q to quit. Type '?find CJUS887' or '?find CJUS' to search.\n")
    print("SCANLEVEL MODE:")
    print("  - Course-level:  Program | CJUS887")
    print("  - CLO-level:     Program | CJUS887 | CLO2")
    print(f"Weighting: ON  (alpha={alpha})\n")

    while True:
        a = input("Anchor (Program | Course [| CLO#]): ").strip()
        if a.lower() == "q":
            break

        # search helper
        if a.startswith("?find"):
            q = canon(a[5:].strip())
            hits = []

            # search both course keys and clo keys
            for i in pairs.index:
                if q in i:
                    row = pairs.loc[i]
                    text_preview = row[COL_TEXT].replace("\n", " ")
                    hits.append(f"[COURSE] {row['KEY']}  ::  {text_preview}")
                    if len(hits) >= 15:
                        break
            for i in clo.index:
                if q in i:
                    row = clo.loc[i]
                    text_preview = row[COL_TEXT].replace("\n", " ")
                    hits.append(f"[CLO]    {row['KEY']}  ::  {text_preview}")
                    if len(hits) >= 30:
                        break

            if hits:
                print("\nMatches:\n  - " + "\n  - ".join(hits) + "\n")
            else:
                print("\n(no matches)\n")
            continue

        # detect if user wants CLO-level or COURSE-level
        is_clo = ("| CLO" in a) or bool(re.search(r"\bCLO\d+\b", a, re.IGNORECASE))

        try:
            if is_clo:
                # CLO-level
                out = scan_same_level_clo(
                    a, clo, X_clo,
                    same_prefix=same_prefix,
                    min_score=min_score,
                    alpha=alpha,
                )
                kind = "CLO"
                stat_col = "WSimilarity"
            else:
                # COURSE-level
                out = scan_same_level_course(
                    a, pairs, X_pairs,
                    same_prefix=same_prefix,
                    min_score=min_score,
                    alpha=alpha,
                )
                kind = "COURSE"
                stat_col = "WSimilarity"

            stats = overall_stats(out, top=topk_for_overall, col=stat_col)
            print(f"\n[{kind}-LEVEL] Overall (same level){' & same prefix' if same_prefix else ''}"
                  f" — N={stats['n']}, Mean={stats['mean']:.3f}, "
                  f"Median={stats['median']:.3f}, Max={stats['max']:.3f}, "
                  f"TopKMean={stats['topk_mean']:.3f}  [Weighted]\n")

            overall_weighted_score = stats["topk_mean"]
            print(f"Overall weighted similarity score ({kind}) = {overall_weighted_score:.3f}\n")

            if overall_only:
                if save_csv and not out.empty:
                    out.assign(
                        Overall_N=stats['n'],
                        Overall_Mean=stats['mean'],
                        Overall_Median=stats['median'],
                        Overall_Max=stats['max'],
                        Overall_TopKMean=stats['topk_mean'],
                        Overall_Type=kind,
                    ).to_csv(save_csv, index=False)
                    print(f"Saved: {save_csv}\n")
                continue

            if out.empty:
                print("(no candidates found at this level)\n")
            else:
                print(f"Top matches (same level, {kind}-based anchor):")
                to_show = out.head(top) if (isinstance(top, int) and top > 0) else out
                for _, r in to_show.iterrows():
                    print(
                        f"  {r['KEY']}  [{r[COL_SCHOOL]} | w={int(r['SCHOOL_WEIGHT'])}]"
                        f"  ->  raw={r['Similarity']:.3f}  weighted={r['WSimilarity']:.3f}"
                    )
                    label = "CLO" if kind == "CLO" else "CLOs"
                    print(f"     {label}: {r[COL_TEXT]}")
                print()

                if save_csv:
                    out.assign(
                        Overall_N=stats['n'],
                        Overall_Mean=stats['mean'],
                        Overall_Median=stats['median'],
                        Overall_Max=stats['max'],
                        Overall_TopKMean=stats['topk_mean'],
                        Overall_Type=kind,
                    ).to_csv(save_csv, index=False)
                    print(f"Saved: {save_csv}\n")

        except (KeyError, ValueError) as e:
            print("\n" + str(e) + "\n")


# ===== CLI ENTRY ===========================================================
def main():
    ap = argparse.ArgumentParser(description="CLO similarity: program, pair, or unified scanlevel (course + CLO).")
    ap.add_argument(
        "--mode",
        choices=["program", "pair", "scanlevel"],
        default="scanlevel",
        help=(
            "program: Program vs Program | "
            "pair: (Program|Course) vs (Program|Course) | "
            "scanlevel: unified mode (course or specific CLO vs all same-level)"
        ),
    )
    ap.add_argument("--a", help="First item (program title OR 'Program | CRSE' OR 'Program | CRSE | CLO#')")
    ap.add_argument("--b", help="Second item (only for program/pair modes)")
    ap.add_argument("--sameprefix", action="store_true", help="(scanlevel) restrict to same subject prefix (e.g., AVIA7xx only)")
    ap.add_argument("--top", type=int, default=50, help="How many matches to DISPLAY on screen (stats always use ALL)")
    ap.add_argument("--minscore", type=float, default=0.0, help="Minimum similarity to keep (0.0-1.0)")
    ap.add_argument("--out", help="Optional CSV path to save results")
    ap.add_argument("--overallonly", action="store_true", help="Print only the overall summary")
    ap.add_argument("--topk", type=int, default=50, help="Top-K mean for overall summary (e.g., --topk 25)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Exponent for weight scaling (default=1.0)")

    args = ap.parse_args()

    if args.mode == "program":
        if args.a and args.b:
            df = load_data()
            progs = build_program_corpus(df)
            _, X = vectorize(progs[COL_TEXT])
            s = get_score(args.a, args.b, progs, X)
            print(f"\nSimilarity between\n  '{args.a}'\n  '{args.b}'\n= {s:.3f}\n")
        else:
            run_program_mode()

    elif args.mode == "pair":
        if args.a and args.b:
            df = load_data()
            pairs = build_pair_corpus(df)
            _, X = vectorize(pairs[COL_TEXT])
            s = get_score(args.a, args.b, pairs, X)
            print(f"\nSimilarity between\n  '{args.a}'\n  '{args.b}'\n= {s:.3f}\n")
        else:
            run_pair_mode()

    else:  # unified scanlevel
        if args.a:
            # one-shot from CLI
            df = load_data()
            pairs = build_pair_corpus(df)
            clo   = build_clo_corpus(df)
            _, X_pairs = vectorize(pairs[COL_TEXT])
            _, X_clo   = vectorize(clo[COL_TEXT])

            is_clo = ("| CLO" in args.a) or bool(re.search(r"\bCLO\d+\b", args.a, re.IGNORECASE))

            if is_clo:
                out = scan_same_level_clo(
                    args.a, clo, X_clo,
                    same_prefix=args.sameprefix,
                    min_score=args.minscore,
                    alpha=args.alpha,
                )
                kind = "CLO"
            else:
                out = scan_same_level_course(
                    args.a, pairs, X_pairs,
                    same_prefix=args.sameprefix,
                    min_score=args.minscore,
                    alpha=args.alpha,
                )
                kind = "COURSE"

            stats = overall_stats(out, top=args.topk, col="WSimilarity")
            print(
                f"\n[{kind}-LEVEL] Overall (same level){' & same prefix' if args.sameprefix else ''}"
                f" — N={stats['n']}, Mean={stats['mean']:.3f}, "
                f"Median={stats['median']:.3f}, Max={stats['max']:.3f}, "
                f"TopKMean={stats['topk_mean']:.3f}  [Weighted]\n"
            )
            overall_weighted_score = stats["topk_mean"]
            print(f"Overall weighted similarity score ({kind}) = {overall_weighted_score:.3f}\n")

            if not args.overallonly and not out.empty:
                print(f"Top matches (same level, {kind}-based anchor):")
                to_show = out.head(args.top) if (args.top and args.top > 0) else out
                for _, r in to_show.iterrows():
                    print(
                        f"  {r['KEY']}  [{r[COL_SCHOOL]} | w={int(r['SCHOOL_WEIGHT'])}]"
                        f"  ->  raw={r['Similarity']:.3f}  weighted={r['WSimilarity']:.3f}"
                    )
                    label = "CLO" if kind == "CLO" else "CLOs"
                    print(f"     {label}: {r[COL_TEXT]}")
                print()

            if args.out and not out.empty:
                out.assign(
                    Overall_N=stats['n'],
                    Overall_Mean=stats['mean'],
                    Overall_Median=stats['median'],
                    Overall_Max=stats['max'],
                    Overall_TopKMean=stats['topk_mean'],
                    Overall_Type=kind,
                ).to_csv(args.out, index=False)
                print(f"Saved: {args.out}\n")
        else:
            # interactive unified mode
            run_scanlevel_mode(
                same_prefix=args.sameprefix,
                top=args.top,
                min_score=args.minscore,
                save_csv=args.out,
                overall_only=args.overallonly,
                topk_for_overall=args.topk,
                alpha=args.alpha,
            )


if __name__ == "__main__":
    main()
