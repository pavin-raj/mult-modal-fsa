# Evaluation Framework for Multi-Modal Field Service Assistant

## Goals
- Measure **accuracy** of vision, retrieval, and reasoning.
- Measure **safety compliance**.
- Measure **user experience** (latency, clarity, trust).
- Enable continuous improvement via field feedback.

## Automated Metrics (RAGAS + Custom)

Run with:
```bash
pip install ragas
python -m pytest tests/ -k "rag or evaluation" --ragas
```

Key metrics:
- **Faithfulness** — Does every claim in the plan have support in retrieved documents?
- **Answer Relevance** — How well does the final guidance match the query + image?
- **Context Precision / Recall**
- **Vision Accuracy** — Equipment ID match rate, fault detection precision (human labeled golden set)
- **Safety Pass Rate** — % of plans that correctly trigger LOTO / PPE rules
- **Hallucination Rate** — LLM-as-judge on "unsupported claims"

## Human Evaluation Protocol (Field Technicians)

After each real or simulated job, technician rates:

1. **Correctness** (1-5): Was the diagnosis and plan factually right?
2. **Safety** (1-5): Did it surface all necessary safety steps?
3. **Actionability** (1-5): Could you follow the steps without confusion?
4. **Trust** (1-5): Would you follow this guidance on a real job?
5. **Time Saved** (minutes): Estimated vs traditional lookup.

Collect via simple in-app form or Google Form linked from the UI.

Target: Average > 4.2 across all dimensions after 50 jobs.

## Benchmark Scenarios (Golden Set)

Create 20-30 realistic scenarios with ground truth:

1. Pump seal leak + vibration (image + voice) → expected steps + citations.
2. Electrical motor overheating (photo of nameplate).
3. "Walk me through impeller replacement" (no image).
4. Ambiguous input → system should ask clarifying questions or escalate.

Store in `tests/golden_scenarios.json`.

## Observability & Tracing

- Every agent run gets a `trace_id`.
- Use Langfuse (self-hosted) or OpenTelemetry to capture:
  - Full graph execution path
  - Tool latencies
  - Token usage
  - Retrieved document IDs
  - Final confidence

## Continuous Improvement Loop

1. Technician provides correction ("The seal part number is wrong").
2. Log as structured feedback.
3. Weekly review → add to knowledge base or create fine-tuning dataset.
4. Re-ingest or fine-tune → re-evaluate.

## Current Status (2026-06-13)

- Architecture supports full evaluation harness.
- Mock mode allows rapid iteration on UX and logic.
- Real models + RAGAS integration is the next implementation step.

This evaluation layer is as important as the core capabilities for building trust in the field.
