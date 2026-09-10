# LLM Evaluation & Benchmarking Report

## Leaderboard (mean overall score, 1-5)

| Rank | Model | Overall | factual_accuracy | relevance | reasoning_quality | hallucination_risk | instruction_adherence | N |
|---|---|---|---|---|---|---|---|---|
| 1 | ModelA | 3.21 | 2.8 | 2.2 | 3.4 | 5.0 | 2.6 | 5 |
| 2 | ModelC | 2.94 | 2.2 | 2.0 | 3.4 | 4.6 | 2.6 | 5 |
| 3 | ModelB | 2.7 | 1.4 | 2.2 | 3.4 | 4.8 | 1.8 | 5 |

## Flagged Responses (13)

| Test Case | Model | Overall | Flags |
|---|---|---|---|
| tc1 | ModelC | 3.05 | possible_hallucination |
| tc2 | ModelA | 3.5 | instruction_miss |
| tc2 | ModelB | 3.05 | instruction_miss |
| tc2 | ModelC | 3.05 | instruction_miss |
| tc3 | ModelA | 2.85 | weak_reasoning |
| tc3 | ModelB | 2.4 | instruction_miss |
| tc3 | ModelC | 2.85 | weak_reasoning |
| tc4 | ModelA | 2.9 | instruction_miss; weak_reasoning |
| tc4 | ModelB | 2.4 | instruction_miss; weak_reasoning |
| tc4 | ModelC | 2.9 | instruction_miss; weak_reasoning |
| tc5 | ModelA | 3.1 | instruction_miss |
| tc5 | ModelB | 2.4 | instruction_miss |
| tc5 | ModelC | 2.85 | instruction_miss |