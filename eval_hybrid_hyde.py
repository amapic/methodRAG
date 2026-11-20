"""
Test de la recherche HYBRIDE (BM25 + dense), de l'hybride + expansion, et de HyDE.

Réutilise eval_query_rewriting.py (même dossier, seule dépendance) : jeux de données, retrievers,
cache LLM des reformulations (llm_cache.json), cache d'embeddings. Seul HyDE demande de nouveaux appels au
LLM (un par requête), mis en cache dans hyde_cache.json.

Lignes du tableau (un tableau par jeu de données) :
  bm25:baseline / bm25:expand        BM25 seul, sans / avec reformulations (expand_concat)
  dense:baseline / dense:expand      dense seul, sans / avec reformulations
  hybrid:baseline                    fusion RRF de bm25:baseline et dense:baseline
  hybrid:expand                      fusion RRF de bm25:expand et dense:expand
  hyde:bm25                          BM25 sur "requête + passage hypothétique écrit par le LLM"
  hyde:dense                         dense sur la moyenne des embeddings (requête, passage hypothétique), comme
                                     dans l'article HyDE (le passage est encodé comme un document, sans préfixe)
  hyde:hybrid                        fusion RRF de hyde:bm25 et hyde:dense

Puis des comparaisons appariées ciblées (écart moyen de NDCG@10, IC 95 % par bootstrap, p par permutation).

Utilisation :
    python eval_hybrid_hyde.py --datasets arxiv_2026 --skip-hyde     # hybride seul : aucun appel LLM
    python eval_hybrid_hyde.py --datasets arxiv_2026 scifact nfcorpus
    python eval_hybrid_hyde.py --max-queries 50                       # essai rapide
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, compare, evaluate

import eval_query_rewriting as base  # même dossier : seul fichier dont ce script dépend

HYDE_CACHE = Path("hyde_cache.json")
HYDE_PROMPT = """Write a short passage (about 80 to 120 words) in the style of a scientific abstract or article \
excerpt that answers or directly addresses the query below. Write it as factual text, without hedging and \
without saying you are unsure, and do not mention the query itself.

Query: {query}

Answer with the passage only."""


# ---------------------------------------------------------------- HyDE
def load_hyde_cache():
    if not HYDE_CACHE.exists():
        return {}
    try:
        return json.loads(HYDE_CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  cache HyDE illisible : on repart d'un cache neuf")
        return {}


def save_hyde_cache(cache):
    HYDE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def generate_hyde(queries, model, cache):
    """Retourne {qid: passage hypothétique}. En cas de refus du LLM, le passage est la requête elle-même."""
    import anthropic
    client = anthropic.Anthropic()
    out, latencies, refused = {}, [], 0
    for qid, q in queries.items():
        key = hashlib.md5(f"hyde|{model}|{q}".encode()).hexdigest()
        if key not in cache:
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": HYDE_PROMPT.format(query=q)}],
                )
            except Exception as e:  # noqa: BLE001
                import anthropic as _a
                raise SystemExit(f"\nERREUR appel LLM ({type(e).__name__}), version anthropic {_a.__version__} :\n{e}\n"
                                 f"Modèle utilisé : {model}")
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            if not text:
                print(f"  /!\\ pas de passage HyDE (stop_reason={resp.stop_reason}) pour la requête {qid} : "
                      "requête d'origine utilisée")
            cache[key] = {"text": text, "latency": time.time() - t0}
            save_hyde_cache(cache)
        passage = cache[key]["text"]
        if not passage:
            refused += 1
            passage = q
        out[qid] = passage
        latencies.append(cache[key]["latency"])
    words = [len(p.split()) for p in out.values()]
    print(f"  HyDE : latence LLM moyenne {statistics.mean(latencies):.2f}s, "
          f"passage moyen {statistics.mean(words):.0f} mots, sans passage (refus) : {refused}/{len(queries)}")
    first = next(iter(queries))
    print(f"  exemple -> requête : {queries[first][:120]!r}\n             passage : {out[first][:300]!r}")
    return out


def hyde_dense_search(de, query_texts, passages, k=base.TOP_K):
    """Recherche dense avec la moyenne des embeddings de la requête (avec préfixe) et du passage (sans préfixe)."""
    qe = de.model.encode([base.QUERY_PREFIX + q for q in query_texts], normalize_embeddings=True)
    he = de.model.encode(passages, normalize_embeddings=True)
    v = qe + he
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norms == 0, 1, norms)
    sims = v @ de.emb.T
    top = np.argsort(-sims, axis=1)[:, :k]
    return [[(de.doc_ids[j], float(sims[i, j])) for j in row] for i, row in enumerate(top)]


# ---------------------------------------------------------------- Statistiques appariées
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


# ---------------------------------------------------------------- Fusion
def fuse(run_list):
    """Fusion RRF de plusieurs runs {qid: {doc: score}}."""
    qids = run_list[0].keys()
    return {
        q: dict(base.rrf_fuse([sorted(r[q].items(), key=lambda x: -x[1]) for r in run_list]))
        for q in qids
    }


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["arxiv_2026", "scifact", "nfcorpus"])
    ap.add_argument("--model", default="claude-haiku-5-5",
                    help="LLM utilisé pour les reformulations (lues dans le cache) et pour HyDE")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--skip-hyde", action="store_true", help="ne teste que l'hybride (aucun appel LLM nouveau)")
    args = ap.parse_args()

    base.LLM_MODEL = args.model  # reformulate() lit cette variable : clé de cache et modèle appelé
    llm_cache = base.load_cache()
    hyde_cache = load_hyde_cache()
    metrics = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
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
        qids = list(queries)
        print(f"  {len(corpus)} documents, {len(queries)} requêtes évaluées, reformulateur : {args.model}")

        doc_ids = list(corpus)
        texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]

        print("  reformulations (depuis le cache LLM)...")
        reforms = base.reformulate(queries, llm_cache)

        bm = base.BM25Retriever(doc_ids, texts)
        de = base.DenseRetriever(doc_ids, texts, ds)
        rb = base.run_methods(bm, queries, reforms)
        rd = base.run_methods(de, queries, reforms)

        runs = {
            "bm25:baseline": rb["baseline"],
            "bm25:expand": rb["expand_concat"],
            "dense:baseline": rd["baseline"],
            "dense:expand": rd["expand_concat"],
            "hybrid:baseline": fuse([rb["baseline"], rd["baseline"]]),
            "hybrid:expand": fuse([rb["expand_concat"], rd["expand_concat"]]),
        }

        if not args.skip_hyde:
            print("  HyDE : génération des passages hypothétiques (les déjà faits viennent du cache)...")
            hyde = generate_hyde(queries, args.model, hyde_cache)
            hb = bm.search([queries[i] + " " + hyde[i] for i in qids])
            hd = hyde_dense_search(de, [queries[i] for i in qids], [hyde[i] for i in qids])
            runs["hyde:bm25"] = base.to_run(qids, hb)
            runs["hyde:dense"] = base.to_run(qids, hd)
            runs["hyde:hybrid"] = fuse([runs["hyde:bm25"], runs["hyde:dense"]])

        q_ranx = Qrels(qrels)
        print(f"\n--- {ds} ---")
        report = compare(q_ranx, [Run(run, name=name) for name, run in runs.items()], metrics=metrics,
                         max_p=0.05, stat_test="student", make_comparable=True)
        print(report)
        report.to_dataframe().to_csv(f"results/hybrid_hyde_{ds}.csv")

        # comparaisons appariées ciblées
        per_q = {name: [ndcg_at_k(qrels[q], run.get(q, {})) for q in qids] for name, run in runs.items()}
        for name, run in runs.items():  # contrôle de cohérence avec ranx
            ref = evaluate(q_ranx, Run(run), "ndcg@10")
            if abs(np.mean(per_q[name]) - ref) > 0.005:
                print(f"  /!\\ NDCG@10 de {name} : {np.mean(per_q[name]):.3f} (ici) contre {ref:.3f} (ranx)")

        wanted = [
            ("hybrid:baseline", "bm25:baseline"), ("hybrid:baseline", "dense:baseline"),
            ("hybrid:baseline", "bm25:expand"),
            ("hybrid:expand", "hybrid:baseline"), ("hybrid:expand", "bm25:expand"), ("hybrid:expand", "dense:expand"),
            ("hyde:bm25", "bm25:expand"), ("hyde:dense", "dense:expand"), ("hyde:hybrid", "hybrid:expand"),
        ]
        pairs = [(a, b) for a, b in wanted if a in runs and b in runs]
        print(f"\n  Comparaisons appariées sur le NDCG@10 ({len(qids)} requêtes) :")
        print(f"  {'A - B':<40}{'écart moyen':>12}{'IC 95 %':>20}{'p':>9}")
        rows = []
        for a, b in pairs:
            mean, lo, hi, p = paired_stats(per_q[a], per_q[b])
            print(f"  {a + ' - ' + b:<40}{mean:>+12.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p:>9.3f}")
            rows.append([a, b, f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{p:.4f}"])
        with open(f"results/hybrid_hyde_{ds}_paired.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["A", "B", "ecart_moyen_ndcg10", "ic95_bas", "ic95_haut", "p_permutation"])
            w.writerows(rows)

    print("\nFini. Lecture : si l'intervalle de confiance d'un écart contient 0, cet écart n'est pas démontré. "
          "Les p-valeurs ne sont pas corrigées pour les comparaisons multiples.")


if __name__ == "__main__":
    main()
