# Quelles méthodes de recherche améliorent vraiment un RAG ?

Banc d'essai reproductible pour mesurer la qualité de la **recherche** (le « R » de RAG) avec et sans reformulation de requête, recherche hybride, HyDE et reranking, sur trois corpus dont un composé d'articles arXiv **postérieurs au seuil de connaissances du LLM**.

*English summary: a reproducible benchmark comparing query expansion, hybrid search (BM25 + dense, RRF), HyDE and cross-encoder reranking on SciFact, NFCorpus and a fresh arXiv corpus (abstracts submitted Aug–Oct 2026, after the LLM's training cutoff). Main findings: enriching the query beats replacing it; hybrid search is a strong, LLM-free default; HyDE helps dense/hybrid retrieval on the two public corpora but is not confirmed on the fresh corpus, and costs about 3 s per query. Results are in French below; scripts and CSV outputs are language-independent.*

Article associé : [lien Medium à ajouter]

## La question

Les tutoriels RAG présentent souvent ces méthodes comme des améliorations quasi automatiques. Ici, chacune est comparée à une baseline sur les mêmes requêtes, avec des tests statistiques appariés et la **latence** indiquée à côté du gain.

On mesure la qualité de la recherche (quels documents remontent), **pas** celle de la réponse rédigée par un LLM.

## Résultats principaux (NDCG@10)

Reformulation avec Claude Haiku 5.5, moteur BM25 :

| Corpus | Requête d'origine | Reformulation seule | Expansion (origine + 3 reformulations) |
|---|---|---|---|
| SciFact (300 requêtes) | 0,662 | 0,596 | **0,716** |
| NFCorpus (323 requêtes) | 0,306 | 0,290 | **0,362** |
| arXiv 2026 (150 questions) | 0,833 | 0,667 | **0,886** |

Toutes configurations :

| Corpus | BM25 | Dense | Hybride | Hybride + expansion | HyDE + hybride |
|---|---|---|---|---|---|
| SciFact | 0,662 | 0,713 | 0,706 | 0,752 | **0,772** |
| NFCorpus | 0,306 | 0,344 | 0,344 | 0,376 | **0,397** |
| arXiv 2026 | 0,833 | 0,807 | 0,895 | 0,886 | **0,905** |

Latence ajoutée par requête, mesurée sur un ordinateur sans GPU (ordres de grandeur) : BM25 moins d'1 ms ; recherche dense 7 à 20 ms ; reformulation par le LLM 1,4 à 2,1 s ; passage HyDE 2,7 à 3,5 s ; reranker (50 candidats) environ 3 s.

Ce qu'on peut en dire, avec les intervalles de confiance des comparaisons appariées :

- **Enrichir la requête aide BM25** sur les trois corpus ; la **remplacer** n'aide pas (et nuit sur SciFact et arXiv).
- **L'hybride bat BM25** de façon démontrée sur les trois corpus (+0,037 à +0,062) sans appel LLM. Face au meilleur des deux moteurs, il l'égale sur SciFact et NFCorpus et le dépasse sur arXiv.
- **HyDE bat l'expansion** avec le moteur dense (SciFact +0,026, NFCorpus +0,028) et avec l'hybride sur NFCorpus (+0,021) ; aucun écart démontré avec BM25, ni sur arXiv.
- **Le reranker testé** (MiniLM, top 50) n'apporte pas de gain régulier : il aide BM25 sur NFCorpus (0,306 → 0,336) mais est neutre ou défavorable ailleurs (par exemple dense + expansion sur SciFact : 0,748 → 0,703).

Détails, tableaux complets et discussion : voir l'article, et les CSV du dossier `results/`.

## Contenu du dépôt

| Fichier | Rôle | Statut |
|---|---|---|
| `eval_query_rewriting.py` | Cœur du banc d'essai : chargement des jeux BEIR, BM25, dense, reformulation (Anthropic), évaluation | exécuté sur données réelles |
| `eval_reranking.py` | Reranking par cross-encoder | exécuté sur données réelles |
| `eval_hybrid_hyde.py` | Recherche hybride et HyDE, comparaisons appariées avec intervalles de confiance | exécuté sur données réelles |
| `eval_reformulators.py` | Compare plusieurs LLM de reformulation dans un même tableau | testé hors ligne uniquement |
| `collect_arxiv.py` | Collecte de résumés arXiv récents (API arXiv) | exécuté sur données réelles |
| `generate_questions.py` | Génère les questions et les jugements de pertinence (Mistral ou Anthropic) | exécuté sur données réelles |
| `check_contamination.py` | Test « sans documents » : le LLM sait-il répondre de mémoire ? Compare à un corpus témoin | testé hors ligne uniquement |
| `export_public_dataset.py` | Exporte le jeu arXiv **sans les résumés** (identifiants, empreintes, questions, qrels) | testé hors ligne uniquement |
| `rebuild_corpus.py` | Reconstruit le corpus arXiv depuis les identifiants et vérifie les empreintes | testé hors ligne uniquement |
| `public_data/` | Version publique du jeu arXiv 2026 (sans les textes des résumés) | — |
| `results/` | Sorties CSV des évaluations | — |

« Testé hors ligne uniquement » signifie vérifié avec de faux serveurs et des données factices, pas encore avec les vraies API.

## Reproduire

Python 3 (testé avec Python 3.13 sous Windows), aucun GPU requis.

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell ; sous Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # puis éditez .env et ajoutez vos clés
```

Puis, dans l'ordre :

```bash
# 1. Baselines, expansion, dense (télécharge SciFact et NFCorpus ; appelle l'API Anthropic pour les reformulations)
python eval_query_rewriting.py --datasets scifact nfcorpus

# 2. Hybride et HyDE
python eval_hybrid_hyde.py --datasets scifact nfcorpus

# 3. Reranking
python eval_reranking.py --datasets scifact nfcorpus

# 4. Corpus arXiv : reconstruction depuis les identifiants publics, puis mêmes évaluations
python rebuild_corpus.py --name arxiv_2026
python eval_query_rewriting.py --datasets arxiv_2026
python eval_hybrid_hyde.py --datasets arxiv_2026
```

Pour refaire la collecte et les questions au lieu de reconstruire (le résultat différera) :

```bash
python collect_arxiv.py --dry-run             # compte les articles disponibles, sans rien écrire
python collect_arxiv.py                       # 1er août au 31 octobre 2026, cs.CL et cs.IR
python generate_questions.py --n 150
```

Contrôle de contamination avec corpus témoin :

```bash
python collect_arxiv.py --start 2024-01-01 --end 2024-03-31 --per-category 500 --name arxiv_2024
python generate_questions.py --name arxiv_2024 --n 150
python check_contamination.py --name arxiv_2024
python check_contamination.py --name arxiv_2026 --compare arxiv_2024
```

**Durées indicatives** (ordinateur sans GPU) : encodage du corpus par `bge-small` environ 9 minutes pour SciFact, 7 pour NFCorpus, 4 pour arXiv 2026 (une seule fois, mis en cache) ; les appels au LLM durent 1,4 à 3,5 s chacun, à raison d'un par requête et par méthode.

**Caches.** Les scripts mémorisent leurs résultats pour ne pas refaire les appels payants : `llm_cache.json` (reformulations), `hyde_cache.json`, `rerank_cache.json`, `questions_cache.json`, `contamination_cache.json` et le dossier `embeddings_cache/`. Ils ne sont pas versionnés (voir `.gitignore`) ; supprimez-les pour tout recalculer. Les sorties d'un LLM changent d'une version de modèle à l'autre : refaire l'expérience avec un autre modèle peut donner des chiffres un peu différents.

## Protocole

- **Corpus.** SciFact (BEIR, 5 183 résumés, 300 requêtes de test), NFCorpus (BEIR, 3 633 documents, 323 requêtes), arXiv 2026 (1 945 résumés soumis entre le 1er août et le 31 octobre 2026, 150 questions générées par LLM à partir des résumés, un document pertinent par question).
- **Moteurs.** BM25 (`bm25s`), dense `BAAI/bge-small-en-v1.5`, fusion par Reciprocal Rank Fusion (k = 60), reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` sur le top 50.
- **LLM.** Reformulations (3 par requête) et passages HyDE (1 par requête) : Claude Haiku 5.5. Questions arXiv : modèle Mistral (`mistral-small-latest` par défaut) ou Anthropic, au choix.
- **Métriques.** NDCG@10 (principale), Recall@10 et @100, MRR@10, via `ranx`.
- **Statistiques.** Test de Student apparié (`ranx`) ; dans `eval_hybrid_hyde.py` et `eval_reformulators.py`, écart moyen de NDCG@10 avec intervalle de confiance à 95 % par bootstrap sur les requêtes et p-valeur par test de permutation.
- **Pourquoi arXiv 2026.** Les articles sont postérieurs à juin 2026, fin des données d'entraînement de Claude Haiku 5.5 d'après la documentation d'Anthropic, pour limiter le risque que le LLM ait mémorisé le corpus. `check_contamination.py` sert à contrôler cette hypothèse.

## Limites

- Anglais uniquement, un seul modèle dense, petit (`bge-small`), un seul reranker.
- Un seul fournisseur de LLM pour la reformulation, et seulement deux modèles de sa gamme essayés.
- Corpus arXiv : questions générées par LLM à partir des résumés (vocabulaire partagé avec les documents, ce qui peut avantager BM25) ; un seul document compté comme pertinent par question, donc des documents qui répondent aussi sont comptés à tort comme faux, ce qui abaisse les scores de toutes les méthodes de la même façon.
- Les p-valeurs ne sont pas corrigées pour les comparaisons multiples : fiez-vous surtout aux intervalles de confiance.
- SciFact et NFCorpus sont publics depuis des années et ont pu être vus par le LLM.
- « Inédit » repose sur les dates de publication tant que le contrôle de contamination n'a pas été mené.
- La qualité de la **réponse finale** n'est pas évaluée.

## Données et licences

- **SciFact et NFCorpus** sont téléchargés depuis la source du benchmark BEIR par les scripts ; ils ne sont pas copiés dans ce dépôt.
- **Résumés arXiv** : ils ne sont **pas redistribués**. `public_data/` ne contient que des identifiants, des empreintes, les questions et les jugements de pertinence ; `rebuild_corpus.py` retélécharge les résumés depuis l'API arXiv. Respectez les conditions d'utilisation de l'API arXiv.
- Les questions et réponses de référence sont des textes générés par des LLM à partir des résumés.

## Problèmes courants sous Windows

- `ImportError: DLL load failed` à l'import de PyTorch ou de `ranx` : installez le *Microsoft Visual C++ Redistributable* x64, puis redémarrez.
- Erreur de certificat SSL au téléchargement de BEIR : le script essaie plusieurs méthodes et contrôle l'empreinte du fichier quand elle est connue.
- Fichier `.env` enregistré sous `.env.txt` par le Bloc-notes : vérifiez l'extension réelle du fichier.

## Références

À vérifier avant publication : BEIR (Thakur et al., 2021), SciFact (Wadden et al., 2020), NFCorpus (Boteva et al., 2016), HyDE (Gao et al., 2022), Reciprocal Rank Fusion (Cormack et al., 2009), BGE (Xiao et al., 2023).

## Licence

MIT, voir `LICENSE`.
