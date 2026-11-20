"""
Génération de questions et de jugements de pertinence (qrels) pour un corpus au format BEIR.

Pour chaque document échantillonné, un LLM écrit UNE question à laquelle le résumé répond. Le document
source est marqué pertinent. Écrit dans datasets/<nom>/ :
  - queries.jsonl           (une question par ligne, format BEIR)
  - qrels/test.tsv          (question -> document source, format BEIR)
  - questions_review.csv    (à relire à la main : question, réponse, document, recouvrement lexical)
  - questions_full.json     (questions + réponses de référence, utile pour évaluer la génération plus tard)

Choix de conception :
  - le modèle générateur (Mistral par défaut, ou Anthropic via --provider anthropic) est différent du modèle
    qui reformule les requêtes (Haiku), pour éviter que le reformulateur ne retravaille des questions écrites
    dans son propre style. Citez le modèle générateur dans l'article ;
  - le LLM doit paraphraser et éviter les noms propres introduits par l'article (méthode, jeu de données),
    pour limiter le biais lexical qui avantagerait BM25 ;
  - le recouvrement lexical question/document est mesuré et affiché ;
  - limite connue : seul le document source est marqué pertinent. D'autres documents peuvent répondre aussi
    et seront comptés à tort comme faux. À signaler dans l'article.

Utilisation :
    python generate_questions.py --n 20            # essai rapide
    python generate_questions.py --n 150           # jeu complet
    python generate_questions.py --n 150 --model ministral-8b-latest   # autre modèle Mistral
    python generate_questions.py --n 150 --provider anthropic          # avec Claude Sonnet

Clés API, dans un fichier .env à côté du script (ou en variables d'environnement) :
    MISTRAL_API_KEY=...        (pour --provider mistral, le défaut)
    ANTHROPIC_API_KEY=...      (pour --provider anthropic)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

DEFAULT_MODELS = {"mistral": "mistral-small-latest", "anthropic": "claude-sonnet-5-5"}
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
PROMPT_VERSION = "v1"
CACHE_FILE = Path("questions_cache.json")

PROMPT = """You are building a test set for a document search engine. Below is the abstract of a recent scientific paper.

Write ONE question that a researcher might type into a search engine when looking for this kind of paper, and that this abstract answers.

Rules:
- The question must be self-contained: never write "this paper", "the authors", "the study" or refer to the abstract.
- Paraphrase: do not copy distinctive phrases from the abstract. Prefer natural, more general wording.
- Do not use the proper names introduced by the paper (new method, model, framework or dataset names); describe what they do or what problem they address instead.
- Be specific enough that only a few papers would answer it, but keep it natural, in one sentence.
- Also give a short reference answer (one or two sentences) that is supported by the abstract.

Return only JSON, nothing else: {{"question": "...", "answer": "..."}}

Abstract:
{abstract}"""

BANNED = re.compile(r"\b(this paper|the paper|the authors?|this study|this work|the abstract|the above)\b", re.I)
STOP = set("""about above after again against also because been before being between both could does doing during each
from further have having here into just more most other over same should some such than that their them then there these
they this those through under until very were what when where which while with would your whose using used based""".split())


def load_env(path=".env"):
    p = Path(__file__).parent / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  cache illisible : on repart d'un cache neuf")
        return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- LLM
class ApiError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def post_json(url, payload, headers, retries=5):
    """POST JSON sans dépendance externe. Réessaie en cas de limite de débit (429) ou d'erreur serveur."""
    import ssl
    import urllib.error
    import urllib.request
    contexts = [ssl.create_default_context()]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    data = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        for ctx in contexts:
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504):
                    last = ApiError(e.code, f"HTTP {e.code} : {body}")
                    break  # on attend puis on réessaie
                raise ApiError(e.code, f"HTTP {e.code} : {body}") from e
            except Exception as e:  # noqa: BLE001  (SSL, réseau...)
                last = e
        time.sleep(2 * (attempt + 1) ** 2)
    raise RuntimeError(f"Échec après {retries} tentatives : {last}")


def ask_mistral(model, abstract):
    """Retourne le texte brut de la réponse de l'API Mistral."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("MISTRAL_API_KEY introuvable. Ajoutez la ligne MISTRAL_API_KEY=votre_clé "
                         "dans le fichier .env à côté du script.")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(abstract=abstract)}],
        "max_tokens": 600,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": "corpus-eval-script/1.0"}
    try:
        data = post_json(MISTRAL_URL, payload, headers)
    except ApiError as e:
        if e.code not in (400, 422):
            raise
        payload.pop("response_format")  # certains modèles refusent le mode JSON : on s'en passe
        data = post_json(MISTRAL_URL, payload, headers)
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):  # certains modèles renvoient une liste de blocs
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


_anthropic_client = None


def ask_anthropic(model, abstract):
    """Retourne le texte brut de la réponse ('' en cas de refus)."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    resp = _anthropic_client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT.format(abstract=abstract)}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def make_asker(provider, model):
    return (lambda abstract: ask_mistral(model, abstract)) if provider == "mistral" \
        else (lambda abstract: ask_anthropic(model, abstract))


def parse_qa(text):
    """Extrait {"question", "answer"} d'une réponse censée être du JSON. Retourne None si invalide."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q, a = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip()
    return {"question": q, "answer": a} if q and a else None


def question_ok(q):
    return q.endswith("?") and 20 <= len(q) <= 300 and not BANNED.search(q)


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in STOP]


def overlap(question, doc_text):
    """Part des mots significatifs de la question qui apparaissent aussi dans le document."""
    q = tokens(question)
    if not q:
        return 0.0
    d = set(tokens(doc_text))
    return sum(1 for t in q if t in d) / len(q)


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arxiv_2026", help="dossier dans datasets/")
    ap.add_argument("--n", type=int, default=150, help="nombre de questions à produire")
    ap.add_argument("--provider", choices=["mistral", "anthropic"], default="mistral",
                    help="fournisseur du modèle qui écrit les questions")
    ap.add_argument("--model", default=None,
                    help="nom du modèle (défaut : mistral-small-latest, ou claude-sonnet-5-5 pour anthropic)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=400, help="longueur minimale du résumé échantillonné")
    ap.add_argument("--max-overlap", type=float, default=None,
                    help="si fourni, écarte les questions dont le recouvrement lexical dépasse cette valeur (0 à 1)")
    ap.add_argument("--force", action="store_true", help="écrase queries.jsonl et qrels/test.tsv existants")
    args = ap.parse_args()

    load_env()
    folder = Path("datasets") / args.name
    corpus_path = folder / "corpus.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"Corpus introuvable : {corpus_path}. Lancez d'abord collect_arxiv.py.")
    if (folder / "queries.jsonl").exists() and not args.force:
        raise SystemExit(f"{folder / 'queries.jsonl'} existe déjà. Utilisez --force pour l'écraser.")

    docs = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
    docs = [d for d in docs if len(d["text"]) >= args.min_chars]
    print(f"{len(docs)} documents éligibles dans {corpus_path}")
    random.Random(args.seed).shuffle(docs)

    model = args.model or DEFAULT_MODELS[args.provider]
    ask = make_asker(args.provider, model)
    print(f"Génération des questions avec {args.provider} / {model}")
    cache = load_cache()

    kept, rejected = [], {"format": 0, "refus_ou_vide": 0, "recouvrement": 0}
    for doc in docs:
        if len(kept) >= args.n:
            break
        key = hashlib.md5(f"{args.provider}|{model}|{PROMPT_VERSION}|{doc['_id']}".encode()).hexdigest()
        if key not in cache:
            try:
                cache[key] = {"raw": ask(doc["text"])}
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                hint = ""
                if getattr(e, "code", None) in (401, 403):
                    hint = "\nClé refusée : vérifiez la clé dans le fichier .env."
                raise SystemExit(f"\nERREUR appel LLM ({type(e).__name__}) avec {args.provider} / {model} :\n{e}{hint}")
            save_cache(cache)

        qa = parse_qa(cache[key]["raw"])
        if qa is None:
            rejected["refus_ou_vide"] += 1
            continue
        if not question_ok(qa["question"]):
            rejected["format"] += 1
            continue
        ov = overlap(qa["question"], doc["title"] + " " + doc["text"])
        if args.max_overlap is not None and ov > args.max_overlap:
            rejected["recouvrement"] += 1
            continue
        kept.append({"doc": doc, "question": qa["question"], "answer": qa["answer"], "overlap": ov})
        if len(kept) % 10 == 0:
            print(f"  {len(kept)}/{args.n} questions retenues")

    if not kept:
        raise SystemExit("Aucune question retenue.")

    (folder / "qrels").mkdir(exist_ok=True)
    with open(folder / "queries.jsonl", "w", encoding="utf-8") as fq, \
            open(folder / "qrels" / "test.tsv", "w", encoding="utf-8", newline="") as ft, \
            open(folder / "questions_review.csv", "w", encoding="utf-8-sig", newline="") as fr:
        tsv = csv.writer(ft, delimiter="\t", lineterminator="\n")
        tsv.writerow(["query-id", "corpus-id", "score"])
        review = csv.writer(fr, delimiter=";")
        review.writerow(["query_id", "question", "reponse_reference", "doc_id", "titre", "recouvrement", "ok_manuel"])
        full = []
        for i, k in enumerate(kept, start=1):
            qid = f"q{i}"
            fq.write(json.dumps({"_id": qid, "text": k["question"]}, ensure_ascii=False) + "\n")
            tsv.writerow([qid, k["doc"]["_id"], 1])
            review.writerow([qid, k["question"], k["answer"], k["doc"]["_id"], k["doc"]["title"],
                             f"{k['overlap']:.2f}", ""])
            full.append({"query_id": qid, "question": k["question"], "answer": k["answer"],
                         "doc_id": k["doc"]["_id"]})
    (folder / "questions_full.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")

    ovs = [k["overlap"] for k in kept]
    print(f"\n{len(kept)} questions écrites dans {folder}")
    print(f"Recouvrement lexical question/document : moyenne {statistics.mean(ovs):.2f}, "
          f"médiane {statistics.median(ovs):.2f}, max {max(ovs):.2f}")
    print(f"Écartées : {rejected}")
    print(f"\nÀ faire maintenant : ouvrez {folder / 'questions_review.csv'} et relisez 30 à 50 questions "
          "(sont-elles naturelles ? le bon document y répond-il ? d'autres documents répondraient-ils aussi ?).")
    print(f"Puis : python eval_query_rewriting.py --datasets {args.name} --no-dense")


if __name__ == "__main__":
    main()
