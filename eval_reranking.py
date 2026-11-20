"""
Test du RERANKING (cross-encoder) sur les jeux BEIR.

Ce fichier réutilise eval_query_rewriting.py (à garder dans le même dossier) :
jeux de données (datasets/), reformulations LLM (llm_cache.json) et embeddings (embeddings_cache/).

Pour chaque retriever (BM25, dense), on compare :
  - baseline                  : requête d'origine, sans reranking
  - baseline+rerank           : top-N de la baseline, réordonné par le cross-encoder
  - expand_concat             : requête enrichie par les reformulations LLM, sans reranking
  - expand_concat+rerank      : top-N de l'expansion, réordonné par le cross-encoder

Le cross-encoder évalue toujours la requête ORIGINALE contre chaque document candidat :
il ne change que l'ordre, pas l'ensemble des candidats.

Utilisation :
    python eval_reranking.py --max-queries 20 --no-dense      # essai rapide
    python eval_reranking.py --datasets scifact nfcorpus      # test complet
"""
import argparse
import json
import time
from pathlib import Path

from ranx import Qrels, Run, compare

import eval_query_rewriting as base  # même dossier : réutilise loaders, retrievers et caches

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # anglais ; pour le français : "BAAI/bge-reranker-v2-m3"
RERANK_CACHE = Path("rerank_cache.json")
LATENCY_KEY = "__ms_per_pair__"


# ---------------------------------------------------------------- Cache des scores du reranker
def load_rr_cache():
    if not RERANK_CACHE.exists():
        return {}
    try:
        return json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  cache de reranking illisible : on repart d'un cache neuf")
        return {}


def save_rr_cache(cache):
    RERANK_CACHE.write_text(json.dumps(cache), encoding="utf-8")


# ---------------------------------------------------------------- Reranking
def doc_text(doc):
    return (doc.get("title", "") + " " + doc.get("text", "")).strip()


def rerank(model, ds, queries, corpus, first_stage, top_n, cache):
    """Réordonne le top-N de chaque requête de first_stage ({qid: {doc: score}}). Retourne un run."""
    candidates, todo = {}, []
    for qid, docs in first_stage.items():
        top = [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])[:top_n]]
        candidates[qid] = top
        for d in top:
            key = f"{ds}|{RERANK_MODEL}|{qid}|{d}"
            if key not in cache:
                todo.append((key, queries[qid], doc_text(corpus[d])))

    if todo:
        print(f"  reranking : {len(todo)} nouvelles paires à scorer (le reste vient du cache)...")
        t0 = time.time()
        scores = model.predict([(q, t) for _, q, t in todo], batch_size=32, show_progress_bar=True)
        cache[LATENCY_KEY] = (time.time() - t0) * 1000 / len(todo)
        for (key, _, _), s in zip(todo, scores):
            cache[key] = float(s)
        save_rr_cache(cache)
    else:
        print("  reranking : tous les scores viennent du cache")

    return {
        qid: {d: cache[f"{ds}|{RERANK_MODEL}|{qid}|{d}"] for d in top}
        for qid, top in candidates.items()
    }


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=50, help="nombre de candidats donnés au reranker")
    ap.add_argument("--no-dense", action="store_true", help="BM25 uniquement (plus rapide)")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    print(f"Chargement du reranker {RERANK_MODEL} ...")
    model = CrossEncoder(RERANK_MODEL, max_length=512)

    llm_cache = base.load_cache()
    rr_cache = load_rr_cache()
    metrics = ["ndcg@10", "recall@10", f"recall@{args.top_n}", "mrr@10"]
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
        texts = [doc_text(corpus[d]) for d in doc_ids]

        print("  reformulations (depuis le cache LLM)...")
        reforms = base.reformulate(queries, llm_cache)

        retrievers = [base.BM25Retriever(doc_ids, texts)]
        if not args.no_dense:
            retrievers.append(base.DenseRetriever(doc_ids, texts, ds))

        q_ranx = Qrels(qrels)
        for r in retrievers:
            print(f"\n--- {ds} / {r.name} ---")
            runs = base.run_methods(r, queries, reforms)

            selected = {}
            for m in ("baseline", "expand_concat"):
                selected[f"{r.name}:{m}"] = runs[m]
                selected[f"{r.name}:{m}+rerank"] = rerank(
                    model, ds, queries, corpus, runs[m], args.top_n, rr_cache)

            if LATENCY_KEY in rr_cache:
                ms = rr_cache[LATENCY_KEY]
                print(f"  coût du reranker : environ {ms:.1f} ms par paire, "
                      f"soit ~{ms * args.top_n / 1000:.1f} s par requête pour {args.top_n} candidats")

            ranx_runs = [Run(run, name=name) for name, run in selected.items()]
            report = compare(q_ranx, ranx_runs, metrics=metrics, max_p=0.05,
                             stat_test="student", make_comparable=True)
            print(report)
            report.to_dataframe().to_csv(f"results/rerank_{ds}_{r.name}.csv")

    print(f"\nFini. Le Recall@{args.top_n} est identique avant/après reranking : "
          "le reranker réordonne les mêmes candidats, il ne peut pas en retrouver de nouveaux.")


if __name__ == "__main__":
    main()
