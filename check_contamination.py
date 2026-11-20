"""
Contrôle de contamination ("closed-book") : le LLM sait-il répondre à vos questions SANS aucun document ?

Principe : on pose à Haiku (le modèle qui reformule dans vos tests) les questions de questions_full.json
sans lui donner le moindre passage, puis un second modèle (le juge, par défaut Mistral) compare chaque réponse
à la réponse de référence : correct / partial / incorrect / abstain (le modèle dit ne pas savoir).

Comment lire le résultat :
  - sur le corpus INÉDIT (articles postérieurs au seuil de connaissances), un taux de réponses correctes faible
    est ce qu'on attend si les documents sont réellement inconnus ;
  - mais ce taux seul ne prouve rien : un taux faible peut venir de questions très précises, un taux non nul peut
    venir de connaissances générales et non d'une mémorisation des articles ;
  - le test devient interprétable en le COMPARANT à un corpus TÉMOIN d'articles plus anciens (que le modèle a
    probablement vus), construit avec les mêmes scripts. Voir --compare.

Étapes pour un corpus témoin (exemple, articles de début 2024) :
    python collect_arxiv.py --start 2024-01-01 --end 2024-03-31 --per-category 500 --name arxiv_2024
    python generate_questions.py --name arxiv_2024 --n 150
    python check_contamination.py --name arxiv_2024
    python check_contamination.py --name arxiv_2026 --compare arxiv_2024

Utilisation :
    python check_contamination.py --name arxiv_2026 --n 20     # essai rapide
    python check_contamination.py --name arxiv_2026

Clés (dans .env à côté du script) : ANTHROPIC_API_KEY (modèle interrogé) et MISTRAL_API_KEY (juge par défaut).
Fichiers produits : datasets/<nom>/contamination_review.csv (à relire à la main) et contamination_results.json.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

DEFAULT_JUDGE = {"mistral": "mistral-small-latest", "anthropic": "claude-sonnet-5-5"}
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
PROMPT_VERSION = "v1"
CACHE_FILE = Path("contamination_cache.json")
VERDICTS = ["correct", "partial", "incorrect", "abstain"]

ANSWER_PROMPT = """Answer the following question about recent scientific research as precisely as you can, \
using only your own knowledge (no documents are provided). If you do not know, say that you do not know \
instead of guessing. Answer in at most three sentences.

Question: {question}"""

JUDGE_PROMPT = """You are grading a candidate answer to a question about a scientific paper.

Question: {question}

Reference answer (taken from the paper's abstract): {reference}

Candidate answer: {candidate}

Choose one verdict:
- "correct": the candidate gives the key content of the reference answer (the wording may differ).
- "partial": the candidate matches only part of the reference, or is generic or vague so that it only partly matches.
- "incorrect": the candidate is wrong, unrelated to the reference, or contradicts it.
- "abstain": the candidate says it does not know or cannot answer.

Return only JSON, nothing else: {{"verdict": "correct|partial|incorrect|abstain", "reason": "<one short sentence>"}}"""


# ---------------------------------------------------------------- Utilitaires
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


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson à 95 % pour une proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------- Appels LLM
class ApiError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def post_json(url, payload, headers, retries=5):
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
                    break
                raise ApiError(e.code, f"HTTP {e.code} : {body}") from e
            except Exception as e:  # noqa: BLE001
                last = e
        time.sleep(2 * (attempt + 1) ** 2)
    raise RuntimeError(f"Échec après {retries} tentatives : {last}")


def ask_mistral(model, prompt, json_mode=False):
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("MISTRAL_API_KEY introuvable. Ajoutez MISTRAL_API_KEY=votre_clé dans le fichier .env.")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 600, "temperature": 0}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": "corpus-eval-script/1.0"}
    try:
        data = post_json(MISTRAL_URL, payload, headers)
    except ApiError as e:
        if not (json_mode and e.code in (400, 422)):
            raise
        payload.pop("response_format")
        data = post_json(MISTRAL_URL, payload, headers)
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


_anthropic_client = None


def ask_anthropic(model, prompt, json_mode=False):
    """Retourne (texte, stop_reason). Un refus donne un texte vide."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    resp = _anthropic_client.messages.create(
        model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def call(provider, model, prompt, json_mode=False):
    try:
        return ask_mistral(model, prompt, json_mode) if provider == "mistral" \
            else ask_anthropic(model, prompt, json_mode)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        hint = "\nClé refusée : vérifiez la clé dans le fichier .env." if getattr(e, "code", None) in (401, 403) else ""
        raise SystemExit(f"\nERREUR appel LLM ({type(e).__name__}) avec {provider} / {model} :\n{e}{hint}")


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    v = str(d.get("verdict", "")).strip().lower()
    return (v if v in VERDICTS else None), str(d.get("reason", "")).strip()


# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arxiv_2026", help="dossier dans datasets/ (doit contenir questions_full.json)")
    ap.add_argument("--n", type=int, default=None, help="nombre de questions à tester (défaut : toutes)")
    ap.add_argument("--answer-model", default="claude-haiku-5-5", help="modèle interrogé sans documents")
    ap.add_argument("--judge-provider", choices=["mistral", "anthropic"], default="mistral")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--compare", default=None, help="autre corpus (déjà testé) à comparer, par exemple un témoin ancien")
    args = ap.parse_args()

    load_env()
    folder = Path("datasets") / args.name
    qfile = folder / "questions_full.json"
    if not qfile.exists():
        raise SystemExit(f"{qfile} introuvable. Lancez d'abord generate_questions.py pour ce corpus.")
    items = json.loads(qfile.read_text(encoding="utf-8"))
    if args.n:
        items = items[: args.n]
    judge_model = args.judge_model or DEFAULT_JUDGE[args.judge_provider]
    print(f"{len(items)} questions de {args.name} | modèle interrogé : {args.answer_model} (sans documents) "
          f"| juge : {args.judge_provider} / {judge_model}")

    cache = load_cache()
    rows, counts, unparsed = [], {v: 0 for v in VERDICTS}, 0
    for i, it in enumerate(items, start=1):
        qid = it["query_id"]
        akey = hashlib.md5(f"answer|{args.answer_model}|{PROMPT_VERSION}|{args.name}|{qid}".encode()).hexdigest()
        if akey not in cache:
            cache[akey] = {"text": call("anthropic", args.answer_model, ANSWER_PROMPT.format(question=it["question"])).strip()}
            save_cache(cache)
        answer = cache[akey]["text"]

        if not answer:  # refus du modèle : compté comme abstention, sans appel au juge
            verdict, reason = "abstain", "le modèle n'a produit aucune réponse (refus)"
        else:
            jkey = hashlib.md5(
                f"judge|{args.judge_provider}|{judge_model}|{PROMPT_VERSION}|{args.name}|{qid}|"
                f"{hashlib.md5(answer.encode()).hexdigest()}".encode()).hexdigest()
            if jkey not in cache:
                raw = call(args.judge_provider, judge_model,
                           JUDGE_PROMPT.format(question=it["question"], reference=it["answer"], candidate=answer),
                           json_mode=True)
                cache[jkey] = {"raw": raw}
                save_cache(cache)
            verdict, reason = parse_verdict(cache[jkey]["raw"])
            if verdict is None:
                unparsed += 1
                continue
        counts[verdict] += 1
        rows.append([qid, it["question"], it["answer"], answer, verdict, reason, ""])
        if i % 10 == 0:
            print(f"  {i}/{len(items)} questions traitées")

    n = sum(counts.values())
    if n == 0:
        raise SystemExit("Aucun verdict exploitable.")

    with open(folder / "contamination_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["query_id", "question", "reponse_reference", "reponse_modele_sans_docs",
                    "verdict_juge", "raison_juge", "verdict_manuel"])
        w.writerows(rows)
    result = {"dataset": args.name, "answer_model": args.answer_model, "judge": f"{args.judge_provider}/{judge_model}",
              "n": n, "counts": counts, "unparsed": unparsed}
    (folder / "contamination_results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    def report(res, title):
        nn, c = res["n"], res["counts"]
        print(f"\n{title} ({nn} questions jugées, modèle : {res['answer_model']}, juge : {res['judge']})")
        for v in VERDICTS:
            lo, hi = wilson(c[v], nn)
            print(f"  {v:<10}{c[v]:>5}  {100 * c[v] / nn:5.1f} %   IC 95 % [{100 * lo:.0f} %, {100 * hi:.0f} %]")
        lo, hi = wilson(c["correct"] + c["partial"], nn)
        print(f"  correct + partial : {100 * (c['correct'] + c['partial']) / nn:.1f} %   IC 95 % [{100 * lo:.0f} %, {100 * hi:.0f} %]")

    report(result, f"Sans documents sur {args.name}")
    if unparsed:
        print(f"  ({unparsed} verdicts du juge illisibles, écartés)")

    if args.compare:
        other_file = Path("datasets") / args.compare / "contamination_results.json"
        if not other_file.exists():
            print(f"\n--compare : {other_file} introuvable. Lancez d'abord check_contamination.py --name {args.compare}.")
        else:
            other = json.loads(other_file.read_text(encoding="utf-8"))
            report(other, f"Sans documents sur {args.compare} (témoin)")
            if other["answer_model"] != result["answer_model"] or other["judge"] != result["judge"]:
                print("  /!\\ modèle interrogé ou juge différents entre les deux corpus : comparaison à éviter.")
            print("\nLecture : si le taux de réponses correctes du corpus inédit est nettement plus bas que celui du témoin "
                  "(intervalles qui ne se chevauchent pas), c'est cohérent avec des documents réellement inconnus du modèle.")

    print(f"\nÀ faire : ouvrez {folder / 'contamination_review.csv'} et relisez 30 à 40 verdicts du juge "
          "(colonne verdict_manuel) : un juge LLM se trompe, et la fiabilité de ce test en dépend.")
    print("Rappel : ce test détecte ce que le modèle sait répondre de mémoire. Il ne prouve ni l'absence ni la "
          "présence des documents dans ses données d'entraînement.")


if __name__ == "__main__":
    main()
