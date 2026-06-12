# Screenshots & figures

Drop the captures here with these exact filenames — the main README already
references them, so they render the moment you commit them.

| File | What to capture | How |
|---|---|---|
| `inference_cli.png` | The agent answering a "both"-intent question with streamed answer, sources and `/timing` breakdown | `python agent/intent_agent.py`, toggle `/timing`, ask a multi-source question |
| `inference_ui.png` | The React frontend mid-answer (optional but strong) | `docker compose up -d` → http://localhost:3000 |
| `graph_stats.png` | Neo4j browser showing the graph (nodes/relationships) or the stats output | http://localhost:7474 → `CALL db.schema.visualization()` — or terminal: `python -m rag.neo4j_search stats` |
| `hammer_status.png` | Evaluation dashboard: grades, failure tags, dataset readiness | `python -m hammer.run_hammer status` |
| `training_run.png` | QLoRA training in progress (loss per step) | terminal during `python training/pipeline.py train --method qlora` |
| `mlflow_runs.png` | MLflow experiment list with runs/metrics, Production model | `mlflow ui --backend-store-uri sqlite:///training/mlflow.db` → http://localhost:5000 |

Tips: dark terminal theme, zoom so text is readable at README width (~900 px),
crop window chrome, and blur anything internal that survived the scrub.
