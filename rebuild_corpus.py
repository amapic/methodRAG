"""
Reconstruit le corpus arXiv à partir de la version publique (public_data/<nom>/), en retéléchargeant les résumés
depuis l'API arXiv par leurs identifiants. Les textes ne sont donc jamais redistribués par ce dépôt.

Utilisation :
    python rebuild_corpus.py --name arxiv_2026

Lit    public_data/<nom>/{doc_ids.tsv, queries.jsonl, qrels/test.tsv}
Écrit  datasets/<nom>/{corpus.jsonl, queries.jsonl, qrels/test.tsv, rebuild_report.txt}

Vérification : chaque résumé téléchargé est comparé à l'empreinte enregistrée à l'export. Un auteur peut avoir
modifié son résumé depuis la collecte d'origine : le rapport indique combien de documents sont identiques,
modifiés ou introuvables. Quelques modifications n'empêchent pas la reproduction, mais citez-les si elles existent.
Respecte la limite d'arXiv (environ une requête toutes les 3 secondes).
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
from pathlib import Path

import collect_arxiv as ca  # même dossier : requêtes HTTP et analyse du flux Atom

BATCH = 50


def doc_hash(title, text):
    normalise = " ".join((title + " " + text).split())  # insensible aux espaces et retours à la ligne
    return hashlib.sha1(normalise.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arxiv_2026")
    args = ap.parse_args()

    src, dst = Path("public_data") / args.name, Path("datasets") / args.name
    ids_file = src / "doc_ids.tsv"
    if not ids_file.exists():
        raise SystemExit(f"{ids_file} introuvable : exécutez ce script à la racine du dépôt.")
    rows = [l.rstrip("\n").split("\t") for l in open(ids_file, encoding="utf-8")][1:]
    ids = [r[0] for r in rows]
    expected = {r[0]: r[1] for r in rows}
    print(f"{len(ids)} documents à reconstruire depuis arXiv")

    fetched = {}
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        url = ca.API + "?" + urllib.parse.urlencode({"id_list": ",".join(chunk), "max_results": len(chunk)})
        docs, _ = ca.parse(ca.http_get(url))
        for d in docs:
            fetched[d["_id"]] = d
        time.sleep(ca.PAUSE)
        print(f"  {min(i + BATCH, len(ids))}/{len(ids)}")

    dst.mkdir(parents=True, exist_ok=True)
    same, changed, missing = 0, [], []
    with open(dst / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id in ids:
            d = fetched.get(doc_id)
            if d is None:
                missing.append(doc_id)
                continue
            if doc_hash(d["title"], d["text"]) == expected[doc_id]:
                same += 1
            else:
                changed.append(doc_id)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    (dst / "qrels").mkdir(exist_ok=True)
    shutil.copy(src / "queries.jsonl", dst / "queries.jsonl")
    shutil.copy(src / "qrels" / "test.tsv", dst / "qrels" / "test.tsv")

    report = [f"Documents attendus : {len(ids)}", f"Identiques à la collecte d'origine : {same}",
              f"Résumé modifié depuis (empreinte différente) : {len(changed)}", f"Introuvables : {len(missing)}"]
    if changed:
        report.append("Modifiés : " + ", ".join(changed))
    if missing:
        report.append("Introuvables : " + ", ".join(missing))
    (dst / "rebuild_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if missing:
        print("/!\\ Des documents manquent : certaines questions peuvent ne plus avoir leur document pertinent.")


if __name__ == "__main__":
    main()
