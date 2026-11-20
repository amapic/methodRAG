"""
Compare plusieurs LLM de REFORMULATION (par défaut Haiku et Sonnet) dans un même tableau.

Réutilise eval_query_rewriting.py (même dossier) : jeux de données, retrievers, cache LLM, cache d'embeddings.
Les reformulations déjà présentes dans llm_cache.json (clé = modèle + requête) ne sont pas redemandées :
seules les combinaisons modèle/requête manquantes appellent l'API.

Pour chaque jeu de données et chaque retriever, le tableau contient :
  - baseline                : requête d'origine
  - <modèle>:expand         : requête d'origine + reformulations (expand_concat)
  - <modèle>:rewrite        : une reformulation remplace la requête
puis des COMPARAISONS APPARIÉES directes (différence moyenne de NDCG@10, intervalle de confiance à 95 %
par bootstrap sur les requêtes, p-valeur par test de permutation par inversion de signes), par exemple
"sonnet:expand - haiku:expand". Les p-valeurs ne sont pas corrigées pour les comparaisons multiples :
fiez-vous surtout à l'intervalle de confiance (s'il contient 0, l'écart n'est pas démontré).

Un diagnostic décrit aussi ce que font les reformulations de chaque modèle (longueur, part des termes de la
requête conservés, termes nouveaux), pour comprendre pourquoi les résultats se ressemblent ou non.

Utilisation :
    python eval_reformulators.py --datasets arxiv_2026 --no-dense
    python eval_reformulators.py --datasets arxiv_2026 scifact nfcorpus
    python eval_reformulators.py --models claude-haiku-5-5 claude-sonnet-5-5 --max-queries 100
"""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, compare, evaluate

import eval_query_rewriting as base  # même dossier

DEFAULT_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
STOP = set("""the and for with from that this these those are was were been being have has had can could would should
how what which who whom whose when where why does did not but into onto over under between about than then there
their them they its our your you any all some such using used based via per""".split())


def short(model):
    return model.replace("claude-", "")


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP]


# ---------------------------------------------------------------- Diagnostic des reformulations
def print_term_stats(models, queries, reforms_by_model):
    print("\n  Ce que font les reformulations de chaque modèle :")
    print(f"  {'modèle':<20}{'mots/reform.':>14}{'termes requête gardés':>24}{'termes nouveaux/requête':>26}"
          f"{'sans reform.':>14}")
    for m in models:
        lens, kept, new, skipped = [], [], [], 0
        for qid, q in queries.items():
            lines = reforms_by_model[m][qid]
            if lines == [q]:  # refus du LLM : pas de reformulation
                skipped += 1
                continue
            qw = set(words(q))
            lens += [len(words(line)) for line in lines]
            if qw:
                kept.append(len(qw & set(words(lines[0]))) / len(qw))  # reformulation n°1 = celle de "rewrite"
            new.append(len(set(w for line in lines for w in words(line)) - qw))
        f = lambda xs: f"{statistics.mean(xs):.1f}" if xs else "-"  # noqa: E731
        k = f"{100 * statistics.mean(kept):.0f} %" if kept else "-"
        print(f"  {short(m):<20}{f(lens):>14}{k:>24}{f(new):>26}{skipped:>14}")
    print("  ('termes requête gardés' = part des mots de la requête retrouvés dans la 1re reformulation, "
          "celle qui remplace la requête dans 'rewrite')")


# ---------------------------------------------------------------- Comparaisons appariées
def ndcg_at_k(rels, ranking, k=10):
    top = sorted(ranking.items(), key=lambda x: -x[1])[:k]
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, (d, _) in enumerate(top))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def paired_stats(a, b, n=10000, seed=0):
    """Différence moyenne a - b, IC95 % bootstrap et p-valeur (permutation par inversion de signes)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n + 1)
    return float(d.mean()), float(lo), float(hi), float(p)


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["arxiv_2026", "scifact", "nfcorpus"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="modèles de reformulation à comparer")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-dense", action="store_true", help="BM25 uniquement (plus rapide)")
    args = ap.parse_args()
    if len(args.models) < 2:
        raise SystemExit("Donnez au moins deux modèles à comparer avec --models")

    llm_cache = base.load_cache()
    Path("results").mkdir(exist_ok=True)

    for ds in args.datasets:
        print(f"\n===== {ds} =====")
        corpus, queries, qrels = base.load_beir(ds)
        queries = {q: t for q, t in queries.items() if q in qrels}
        qrels = {q: qrels[q] for q in queries}
        if args.max_queries:
            keep = list(queries)[: args.max_queries]
            queries = {q: queries[q] for q in keep}
            qrels = {q: qrels[q] for q in keep}
        if not queries:
            raise RuntimeError(f"Aucune requête avec qrels pour {ds}")
        print(f"  {len(corpus)} documents, {len(queries)} requêtes évaluées")

        doc_ids = list(corpus)
        texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]

        reforms_by_model = {}
        for m in args.models:
            print(f"  reformulations avec {m} (les déjà faites viennent du cache)...")
            base.LLM_MODEL = m  # reformulate() lit cette variable du module : clé de cache et modèle appelé
            reforms_by_model[m] = base.reformulate(queries, llm_cache)
        print_term_stats(args.models, queries, reforms_by_model)

        retrievers = [base.BM25Retriever(doc_ids, texts)]
        if not args.no_dense:
            retrievers.append(base.DenseRetriever(doc_ids, texts, ds))

        q_ranx = Qrels(qrels)
        qids = list(queries)
        for r in retrievers:
            print(f"\n--- {ds} / {r.name} ---")
            runs = {}
            for i, m in enumerate(args.models):
                mr = base.run_methods(r, queries, reforms_by_model[m])
                if i == 0:
                    runs[f"{r.name}:baseline"] = mr["baseline"]
                runs[f"{r.name}:{short(m)}:expand"] = mr["expand_concat"]
                runs[f"{r.name}:{short(m)}:rewrite"] = mr["rewrite"]

            ranx_runs = [Run(run, name=name) for name, run in runs.items()]
            report = compare(q_ranx, ranx_runs, metrics=METRICS, max_p=0.05,
                             stat_test="student", make_comparable=True)
            print(report)
            report.to_dataframe().to_csv(f"results/reformulators_{ds}_{r.name}.csv")

            # NDCG@10 par requête (calculé ici) pour les comparaisons appariées directes
            per_q = {name: [ndcg_at_k(qrels[q], run.get(q, {})) for q in qids] for name, run in runs.items()}
            for name, run in runs.items():  # contrôle de cohérence avec ranx
                ref = evaluate(q_ranx, Run(run), "ndcg@10")
                if abs(np.mean(per_q[name]) - ref) > 0.005:
                    print(f"  /!\\ NDCG@10 de {name} : {np.mean(per_q[name]):.3f} (ici) contre {ref:.3f} (ranx)")

            pairs = [(f"{r.name}:{short(m)}:expand", f"{r.name}:baseline") for m in args.models]
            for j in range(1, len(args.models)):
                for i in range(j):
                    mj, mi = short(args.models[j]), short(args.models[i])
                    pairs.append((f"{r.name}:{mj}:expand", f"{r.name}:{mi}:expand"))
                    pairs.append((f"{r.name}:{mj}:rewrite", f"{r.name}:{mi}:rewrite"))

            print(f"\n  Comparaisons appariées sur le NDCG@10 ({len(qids)} requêtes) :")
            print(f"  {'A - B':<52}{'écart moyen':>12}{'IC 95 %':>20}{'p':>9}")
            rows = []
            for a, b in pairs:
                mean, lo, hi, p = paired_stats(per_q[a], per_q[b])
                print(f"  {a + ' - ' + b:<52}{mean:>+12.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p:>9.3f}")
                rows.append([a, b, f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{p:.4f}"])
            with open(f"results/reformulators_{ds}_{r.name}_paired.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["A", "B", "ecart_moyen_ndcg10", "ic95_bas", "ic95_haut", "p_permutation"])
                w.writerows(rows)

    print("\nFini. Lecture : si l'intervalle de confiance d'un écart contient 0, cet écart n'est pas démontré. "
          "Les p-valeurs ne sont pas corrigées pour les comparaisons multiples.")


if __name__ == "__main__":
    main()
