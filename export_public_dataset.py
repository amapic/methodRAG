"""
Prépare la version PUBLIABLE du corpus arXiv : identifiants + empreintes des résumés, questions et jugements
de pertinence, SANS les textes des résumés (que vous n'avez pas à redistribuer).

À lancer UNE FOIS chez vous, avant de pousser le dépôt :
    python export_public_dataset.py --name arxiv_2026

Lit    datasets/<nom>/{corpus.jsonl, queries.jsonl, qrels/test.tsv, meta.json}
Écrit  public_data/<nom>/{doc_ids.tsv, queries.jsonl, qrels/test.tsv, meta.json}

doc_ids.tsv contient pour chaque document son identifiant arXiv et une empreinte (SHA-1 tronqué) de
"titre + résumé" : rebuild_corpus.py s'en sert pour vérifier que le corpus reconstruit est identique.

Les réponses de référence (questions_full.json) ne sont pas copiées par défaut ; ajoutez --with-answers
pour les inclure (ce sont des textes générés à partir des résumés : à vous de juger si vous les publiez).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def doc_hash(title, text):
    normalise = " ".join((title + " " + text).split())  # insensible aux espaces et retours à la ligne
    return hashlib.sha1(normalise.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arxiv_2026")
    ap.add_argument("--with-answers", action="store_true", help="copie aussi questions_full.json")
    args = ap.parse_args()

    src, dst = Path("datasets") / args.name, Path("public_data") / args.name
    for needed in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        if not (src / needed).exists():
            raise SystemExit(f"{src / needed} introuvable. Lancez d'abord collect_arxiv.py et generate_questions.py.")

    (dst / "qrels").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src / "corpus.jsonl", encoding="utf-8") as f, \
            open(dst / "doc_ids.tsv", "w", encoding="utf-8", newline="") as out:
        out.write("arxiv_id\tsha1_titre_resume\n")
        for line in f:
            d = json.loads(line)
            out.write(f"{d['_id']}\t{doc_hash(d['title'], d['text'])}\n")
            n += 1
    shutil.copy(src / "queries.jsonl", dst / "queries.jsonl")
    shutil.copy(src / "qrels" / "test.tsv", dst / "qrels" / "test.tsv")
    for optional in ["meta.json"] + (["questions_full.json"] if args.with_answers else []):
        if (src / optional).exists():
            shutil.copy(src / optional, dst / optional)
    print(f"{n} documents exportés (identifiants et empreintes uniquement) vers {dst}")
    print("Aucun texte de résumé n'a été copié. Vous pouvez versionner le dossier public_data/.")


if __name__ == "__main__":
    main()
