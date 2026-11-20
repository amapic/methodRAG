"""
Test de l'efficacité du Query Rewriting / Expansion sur des jeux BEIR.

Compare, pour BM25 et pour un retriever dense :
  - baseline      : requête d'origine
  - rewrite       : une reformulation LLM remplace la requête
  - expand_concat : requête d'origine + reformulations concaténées en une seule requête
  - multi_rrf     : chaque variante est recherchée séparément, puis fusion RRF

Installation :
    pip install bm25s ranx anthropic numpy            # BM25 seul
    pip install sentence-transformers                 # en plus, pour le retriever dense (PyTorch)

Utilisation :
    export ANTHROPIC_API_KEY=...
    python eval_query_rewriting.py --datasets scifact nfcorpus
    python eval_query_rewriting.py --datasets scifact --max-queries 50   # test rapide
"""
import argparse
import csv
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, compare

import bm25s

def load_env(path=".env"):
    """Lit un fichier .env (lignes CLE=valeur) à côté du script, sans écraser les variables déjà définies."""
    p = Path(__file__).parent / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

LLM_MODEL = "claude-haiku-5-5"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"  # corpus anglais ; pour le français : "BAAI/bge-m3"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_REFORMULATIONS = 3
TOP_K = 100
RRF_K = 60
CACHE_FILE = Path("llm_cache.json")
DATA_DIR = Path("datasets")
EMB_DIR = Path("embeddings_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # supprime l'avertissement Windows

PROMPT = """You help improve document search. Rewrite the search query below into {n} different \
reformulations that would help retrieve relevant documents: use synonyms, more precise or more \
technical vocabulary, and spell out abbreviations or implicit context. Keep the original meaning.

Query: {query}

Answer with exactly {n} lines, one reformulation per line, no numbering, no commentary."""


# ---------------------------------------------------------------- Téléchargement
KNOWN_MD5 = {"scifact": "5f7d1de60b170fc8027bb7898e2efca1"}  # md5 du README officiel BEIR


def download(url, dest):
    """Télécharge url vers dest. Essaie : certificats système, certifi, puis sans vérification SSL."""
    import ssl
    contexts = [("certificats système", ssl.create_default_context())]
    try:
        import certifi
        contexts.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    contexts.append(("SANS vérification SSL", unverified))

    last = None
    for label, ctx in contexts:
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            if label.startswith("SANS"):
                print("  /!\\ téléchargé sans vérification du certificat SSL (intégrité vérifiée par md5 si connu)")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  échec avec {label} : {type(e).__name__}")
    raise last


# ---------------------------------------------------------------- Chargement BEIR (sans torch)
def load_beir(name, split="test"):
    """Télécharge (une seule fois) et charge un jeu BEIR. Retourne corpus, queries, qrels."""
    DATA_DIR.mkdir(exist_ok=True)
    folder = DATA_DIR / name
    if not folder.exists():
        zip_path = DATA_DIR / f"{name}.zip"
        if not zip_path.exists():
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
            print(f"  téléchargement de {url} ...")
            download(url, zip_path)
            md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
            expected = KNOWN_MD5.get(name)
            if expected and md5 != expected:
                zip_path.unlink()
                raise RuntimeError(f"md5 incorrect pour {name}.zip ({md5}), fichier supprimé")
            print(f"  md5 {md5}" + (" (vérifié)" if expected else " (non vérifié, md5 de référence inconnu)"))
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)

    corpus, queries, qrels = {}, {}, {}
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # en-tête
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return corpus, queries, qrels


# ---------------------------------------------------------------- LLM + cache
def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  cache illisible (vide ou corrompu) : on repart d'un cache neuf")
        return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def reformulate(queries, cache):
    """Retourne {qid: [reformulations]} ; résultats mis en cache (reproductibilité + coût)."""
    import anthropic
    client = anthropic.Anthropic()
    out, latencies = {}, []
    for qid, q in queries.items():
        key = hashlib.md5(f"{LLM_MODEL}|{N_REFORMULATIONS}|{q}".encode()).hexdigest()
        if key not in cache:
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": PROMPT.format(n=N_REFORMULATIONS, query=q)}],
                )
            except Exception as e:  # noqa: BLE001
                import anthropic as _a
                raise SystemExit(
                    f"\nERREUR appel LLM ({type(e).__name__}), version anthropic {_a.__version__} :\n{e}\n"
                    f"Modèle utilisé : {LLM_MODEL}"
                )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not lines:
                print(f"  /!\\ pas de reformulation (stop_reason={resp.stop_reason}) pour la requête "
                      f"{qid} : requête d'origine conservée")
            cache[key] = {"lines": lines[:N_REFORMULATIONS], "latency": time.time() - t0}
            save_cache(cache)
        out[qid] = cache[key]["lines"] or [q]
        latencies.append(cache[key]["latency"])
    n_empty = sum(1 for qid in queries if out[qid] == [queries[qid]])
    print(f"  latence LLM moyenne par requête : {np.mean(latencies):.2f}s")
    print(f"  requêtes sans reformulation (refus du LLM) : {n_empty}/{len(queries)}")
    return out


# ---------------------------------------------------------------- Retrievers
class BM25Retriever:
    name = "bm25"

    def __init__(self, doc_ids, texts):
        self.doc_ids = doc_ids
        self.ret = bm25s.BM25()
        self.ret.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

    def search(self, queries, k=TOP_K):
        toks = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        idx, scores = self.ret.retrieve(toks, k=k, show_progress=False)
        return [[(self.doc_ids[i], float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(idx, scores)]


class DenseRetriever:
    name = "dense"

    def __init__(self, doc_ids, texts, dataset):
        from sentence_transformers import SentenceTransformer  # import ici : torch seulement si dense
        self.doc_ids = np.array(doc_ids)
        self.model = SentenceTransformer(DENSE_MODEL)

        # cache disque : un fichier par (jeu de données, modèle, empreinte du corpus)
        EMB_DIR.mkdir(exist_ok=True)
        fingerprint = hashlib.md5(
            ("|".join(map(str, doc_ids)) + f"|{len(texts)}|{sum(len(t) for t in texts)}").encode()
        ).hexdigest()[:10]
        path = EMB_DIR / f"{dataset}_{DENSE_MODEL.replace('/', '_')}_{fingerprint}.npy"
        if path.exists():
            self.emb = np.load(path)
            print(f"  embeddings du corpus chargés depuis {path} (pas de réencodage)")
        else:
            print("  encodage du corpus (peut être long sur CPU)...")
            self.emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=True)
            np.save(path, self.emb)
            print(f"  embeddings sauvegardés dans {path}")

    def search(self, queries, k=TOP_K):
        q = self.model.encode([QUERY_PREFIX + x for x in queries], normalize_embeddings=True)
        sims = q @ self.emb.T
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[(self.doc_ids[j], float(sims[i, j])) for j in row] for i, row in enumerate(top)]


# ---------------------------------------------------------------- Méthodes
def to_run(qids, results):
    return {qid: {d: s for d, s in res} for qid, res in zip(qids, results)}


def rrf_fuse(rankings, k=RRF_K):
    fused = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:TOP_K]


def run_methods(retriever, queries, reforms):
    qids = list(queries)
    runs = {}

    t0 = time.time()
    base = retriever.search([queries[i] for i in qids])
    base_time = (time.time() - t0) / len(qids)
    runs["baseline"] = to_run(qids, base)

    rew = retriever.search([reforms[i][0] for i in qids])
    runs["rewrite"] = to_run(qids, rew)

    # variantes uniques par requête (si le LLM a refusé, il ne reste que la requête d'origine)
    all_variants = {i: list(dict.fromkeys([queries[i]] + reforms[i])) for i in qids}

    concat = retriever.search([" ".join(all_variants[i]) for i in qids])
    runs["expand_concat"] = to_run(qids, concat)

    # multi-requêtes : toutes les variantes recherchées séparément puis fusionnées
    flat, owner = [], []
    for i in qids:
        for v in all_variants[i]:
            flat.append(v)
            owner.append(i)
    flat_res = retriever.search(flat)
    per_q = {i: [] for i in qids}
    for i, res in zip(owner, flat_res):
        per_q[i].append(res)
    runs["multi_rrf"] = {i: dict(rrf_fuse(per_q[i])) for i in qids}

    print(f"  temps de recherche par requête (baseline, hors LLM) : {base_time*1000:.1f} ms")
    return runs


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-dense", action="store_true", help="BM25 uniquement (plus rapide)")
    args = ap.parse_args()

    cache = load_cache()
    metrics = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
    Path("results").mkdir(exist_ok=True)

    for ds in args.datasets:
        print(f"\n===== {ds} =====")
        corpus, queries, qrels = load_beir(ds)

        # queries.jsonl contient aussi les requêtes train : on garde seulement celles du split test
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

        print("  reformulation des requêtes...")
        reforms = reformulate(queries, cache)

        retrievers = [BM25Retriever(doc_ids, texts)]
        if not args.no_dense:
            retrievers.append(DenseRetriever(doc_ids, texts, ds))

        q_ranx = Qrels(qrels)
        for r in retrievers:
            print(f"\n--- {ds} / {r.name} ---")
            runs = run_methods(r, queries, reforms)
            ranx_runs = [Run(run, name=f"{r.name}:{m}") for m, run in runs.items()]
            # test de significativité (t-test apparié, correction de Bonferroni) vs 1er run = baseline
            report = compare(q_ranx, ranx_runs, metrics=metrics, max_p=0.05,
                             stat_test="student", make_comparable=True)
            print(report)
            report.to_dataframe().to_csv(f"results/{ds}_{r.name}.csv")

    print("\nFini. Tableaux CSV dans ./results/ ; lettres en exposant = significativement "
          "meilleur que la méthode correspondante (p < 0.05).")


if __name__ == "__main__":
    main()
