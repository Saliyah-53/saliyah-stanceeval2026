# saliyah at StanceEval-2026 — Arabic Stance Detection

Code and notebooks for our submission to **StanceEval-2026** (Arabic stance
detection over the Mawqif-v2 corpus, ArabicNLP @ EMNLP 2026), on **both tracks**.

**Final official systems (blind test, Favg2):**

| Track | System | Notebook | Favg2 | Rank |
|-------|--------|----------|:-----:|:----:|
| Track 1 (Seen) | ALLaM + Fanar fusion, guidance prompt | `final_systems/SaliAI_DeepPrompt_Fusion_Track1.ipynb` | **0.747** | 18 |
| Track 2 (Unseen) | Fanar-9B, guidance 3-shot prompt | `final_systems/SaliAI_DeepPrompt_Fusion_Track2.ipynb` | **0.834** | 13 |

We treat the task as a controlled comparison of two paradigms — fine-tuned Arabic
encoders vs. **few-shot (3-shot)** native Arabic LLMs — and report every
configuration we tried, including negative results. The exact prompts (Arabic
originals) are inside the notebooks; English glosses are in the paper appendix.

---

## Repository structure

```
saliyah-stanceeval2026/
├── README.md
├── final_systems/       # the two submitted systems (0.747 / 0.834)
├── encoders/            # multi-task encoders, preprocessing, target masking
├── llms/                # few-shot LLM runs (Fanar, ALLaM, SILMA, Jais), CoT, self-consistency
├── fusion/              # ALLaM+Fanar fusion, triple fusion
└── negative_results/    # fine-tuning, distillation, synthetic aug., hashtag rules, complex prompting
```

---

## Notebook → result map

**Final systems**

| Notebook | Favg2 |
|---|:--:|
| `final_systems/SaliAI_DeepPrompt_Fusion_Track1.ipynb` (ALLaM+Fanar fusion) | **0.747** |
| `final_systems/SaliAI_DeepPrompt_Fusion_Track2.ipynb` (Fanar guidance) | **0.834** |

**Few-shot LLMs**

| Notebook | Note | Favg2 |
|---|---|:--:|
| `llms/SaliAI_LLM_Compare4_Track2.ipynb` | Fanar / SILMA / Jais / ALLaM (Track 2) | Fanar 0.828 |
| `llms/SaliAI_ALLaM_CoT_Vote_Track1.ipynb` | ALLaM CoT + self-consistency | 0.726 |
| `llms/SaliAI_Fanar_Track1.ipynb` | Fanar few-shot (Track 1) | 0.715 |

**Fusion**

| Notebook | Note | Favg2 |
|---|---|:--:|
| `fusion/SaliAI_TripleFusion_Track1.ipynb` | + SILMA (weaker member hurts) | 0.717 |
| `fusion/SaliAI_Fanar_SILMA_Fusion_Track1.ipynb` | Fanar + SILMA fusion | — |

**Encoders (multi-task learning)**

| Notebook | Note | Favg2 |
|---|---|:--:|
| `encoders/SaliAI_Track2_final.ipynb` | best encoder, Track 2 (AraBERT-large) | 0.694 |
| `encoders/SaliAI_Track1_final.ipynb` | best trained, Track 1 | 0.685 |

**Negative results (transparency)**

| Notebook | Idea | Favg2 |
|---|---|:--:|
| `negative_results/SaliAI_ALLaM_QLoRA.ipynb` | LLM fine-tuning | 0.412 |
| `negative_results/SaliAI_Track1_Distillation.ipynb` | transductive distillation | 0.615 |
| `negative_results/SaliAI_Track1_Synthetic_Fanar.ipynb` | synthetic augmentation | 0.649 |
| `negative_results/SaliAI_Hashtag_Stance_Track1.ipynb` | hashtag-override rule | 0.658 |
| `negative_results/SaliAI_3Ideas_*` | mixture-of-experts / sarcasm / debate | ≤0.740 |

---

## How to run

1. Open a notebook in **Google Colab** and enable a **T4 GPU**.
2. Place the shared-task data (`test_seen.csv`, `test_unseen.csv`, and the
   train/dev files) in the paths noted at the top of each notebook.
3. LLMs load in **4-bit** (`bitsandbytes`); each notebook writes a `submission.zip`
   ready for Codabench.

Dependencies: `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece`,
`pandas`.

## Prompts

The exact Arabic prompts (the unified and the guidance-rich instructions plus the
three fixed few-shot examples) are defined inside the LLM notebooks; English
glosses are provided in the paper appendix.

## Data

We use the **Mawqif-v2** corpus released by the StanceEval-2026 organizers. This
repository does **not** redistribute the data or the blind-test gold labels.
