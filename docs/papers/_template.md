# <Paper title>

**Authors — Venue Year** · Paper: <link> · Code here: `llm_routing/routers/<name>/`

## 1. The problem it addresses
What is wrong or missing in earlier routers? (1–2 paragraphs)

## 2. Key idea in one picture
ASCII diagram of the decision flow: what the router sees, what it outputs, which model answers.

## 3. Method
- What is learned (target) and from which labels
- Router architecture and loss
- How the cost/quality knob works at test time
- Paper-specific tricks (with the equation and where it lives in our code)

## 4. Router card
Fill in the row for [docs/comparison.md](../comparison.md): decision time, candidates, learns, labels, k samples, model, knob, overhead.

## 5. Paper setup vs. our implementation
| | paper | ours |
|---|---|---|
| data | | |
| models | | |
| router encoder / hyperparameters | | |
| deviations and why | | |

## 6. Results on our benchmarks
Link the leaderboard rows; discuss what the curve shows compared with baselines and other routers.

## 7. Limitations / open questions
