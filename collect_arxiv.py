"""
Collecte de résumés arXiv récents pour constituer un corpus "inédit" (postérieur au seuil de connaissances du LLM).

Le corpus est écrit au format BEIR dans datasets/<nom>/corpus.jsonl, directement lisible par
eval_query_rewriting.py / eval_reranking.py une fois les requêtes et les qrels ajoutés
(étape suivante : génération de questions).

Aucune bibliothèque à installer (bibliothèque standard uniquement).

Exemples :
    python collect_arxiv.py --dry-run                                  # compte seulement, sans rien écrire
    python collect_arxiv.py                                            # cs.CL + cs.IR, 1er août au 31 oct 2026
    python collect_arxiv.py --categories cs.CL cs.IR cs.LG --per-category 1000
    python collect_arxiv.py --start 2026-09-01 --end 2026-10-31 --name arxiv_sept_oct

Règles de politesse arXiv : ~1 requête toutes les 3 secondes (respecté ici). Le script identifie
ses requêtes par un User-Agent. Vérifiez les conditions d'utilisation de l'API avant de redistribuer le corpus.
"""
import argparse
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

API = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
USER_AGENT = "corpus-eval-script/1.0 (usage personnel, evaluation de recherche)"
PAUSE = 3.2  # secondes entre deux requêtes
PAGE = 100  # résultats par requête


# ---------------------------------------------------------------- Réseau
def http_get(url, retries=5):
    """GET avec certificats système puis certifi, et nouvelles tentatives si arXiv est saturé (429/503)."""
    import ssl
    contexts = [ssl.create_default_context()]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass

    last = None
    for attempt in range(retries):
        for ctx in contexts:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 503):
                    break  # on attend puis on réessaie
                raise
            except Exception as e:  # noqa: BLE001  (SSL, réseau...)
                last = e
        wait = PAUSE * (attempt + 2) * 2
        print(f"    échec ({type(last).__name__}), nouvelle tentative dans {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"Échec de la requête après {retries} tentatives : {last}")


# ---------------------------------------------------------------- Analyse de la réponse
def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def parse(xml_bytes):
    """Retourne (liste de documents, nombre total de résultats annoncé par arXiv)."""
    root = ET.fromstring(xml_bytes)
    total = int(root.findtext("os:totalResults", "0", NS))
    docs = []
    for e in root.findall("a:entry", NS):
        raw_id = e.findtext("a:id", "", NS)
        if "/api/errors" in raw_id:
            raise RuntimeError(f"Erreur renvoyée par l'API arXiv : {clean(e.findtext('a:summary', '', NS))}")
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1]).replace("/", "_")
        prim = e.find("arxiv:primary_category", NS)
        docs.append({
            "_id": arxiv_id,
            "title": clean(e.findtext("a:title", "", NS)),
            "text": clean(e.findtext("a:summary", "", NS)),
            "metadata": {
                "published": e.findtext("a:published", "", NS),  # date de soumission de la v1
                "primary_category": prim.get("term") if prim is not None else "",
            },
        })
    return docs, total


def is_valid(doc, min_chars):
    t = doc["text"].lower()
    if len(doc["text"]) < min_chars:
        return False
    if "withdrawn" in t[:300] or "has been withdrawn" in t:
        return False
    return True


# ---------------------------------------------------------------- Requêtes par fenêtres de dates
def windows(start, end, days):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def build_url(category, ws, we, start_idx, n):
    # submittedDate = date de soumission de la version 1 (et non de la dernière mise à jour)
    query = f"cat:{category} AND submittedDate:[{ws:%Y%m%d}0000 TO {we:%Y%m%d}2359]"
    params = {
        "search_query": query,
        "start": start_idx,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    return API + "?" + urllib.parse.urlencode(params)


def collect(categories, start, end, per_category, window_days, min_chars, dry_run):
    wins = list(windows(start, end, window_days))
    quota = math.ceil(per_category / len(wins))
    print(f"{len(wins)} fenêtres de {window_days} jours, quota de {quota} documents par fenêtre et par catégorie")

    seen, docs = set(), []
    for cat in categories:
        for ws, we in wins:
            kept, idx, total = 0, 0, None
            while kept < quota and (total is None or idx < total):
                n = min(PAGE, quota - kept) if not dry_run else 1
                batch, total = parse(http_get(build_url(cat, ws, we, idx, n)))
                time.sleep(PAUSE)
                if dry_run or not batch:
                    break
                idx += len(batch)
                for d in batch:
                    if d["_id"] in seen or not is_valid(d, min_chars):
                        continue
                    seen.add(d["_id"])
                    docs.append(d)
                    kept += 1
                    if kept >= quota:
                        break
            print(f"  {cat} {ws} -> {we} : "
                  + (f"{total} articles disponibles" if dry_run else f"{kept} gardés (sur {total} disponibles)"))
    return docs


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser(description="Collecte de résumés arXiv récents au format BEIR")
    ap.add_argument("--categories", nargs="+", default=["cs.CL", "cs.IR"])
    ap.add_argument("--start", default="2026-08-01", help="date de soumission minimale (AAAA-MM-JJ)")
    ap.add_argument("--end", default="2026-10-31", help="date de soumission maximale (AAAA-MM-JJ)")
    ap.add_argument("--per-category", type=int, default=1000, help="nombre cible de documents par catégorie")
    ap.add_argument("--window-days", type=int, default=7, help="taille des fenêtres pour répartir l'échantillon dans le temps")
    ap.add_argument("--min-chars", type=int, default=300, help="longueur minimale du résumé")
    ap.add_argument("--name", default="arxiv_2026", help="nom du dossier dans datasets/")
    ap.add_argument("--dry-run", action="store_true", help="interroge arXiv pour compter, sans rien écrire")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        raise SystemExit("--start doit précéder --end")
    print(f"Période : {start} -> {end} | catégories : {', '.join(args.categories)}")

    docs = collect(args.categories, start, end, args.per_category, args.window_days,
                   args.min_chars, args.dry_run)
    if args.dry_run:
        print("\nMode --dry-run : rien n'a été écrit.")
        return
    if not docs:
        raise SystemExit("Aucun document collecté : vérifiez les dates et les catégories.")

    out = Path("datasets") / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    meta = {
        "source": "arXiv API (https://export.arxiv.org/api/query)",
        "collected_on": date.today().isoformat(),
        "categories": args.categories,
        "submitted_from": start.isoformat(),
        "submitted_to": end.isoformat(),
        "n_docs": len(docs),
        "min_chars": args.min_chars,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    lengths = [len(d["text"]) for d in docs]
    print(f"\n{len(docs)} documents écrits dans {out / 'corpus.jsonl'}")
    print(f"Longueur moyenne des résumés : {sum(lengths) // len(lengths)} caractères")
    print("Par catégorie principale :", dict(Counter(d["metadata"]["primary_category"] for d in docs).most_common(8)))
    print("\nÉtape suivante : générer les questions et les jugements de pertinence (queries.jsonl et qrels/test.tsv).")


if __name__ == "__main__":
    main()
