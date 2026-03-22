# Pharmacy Smart Sync Scheduler

A production-quality Streamlit application simulating a queue-based stock
synchronisation system for a multi-branch pharmacy network, with schema
normalisation, timezone handling, conflict resolution, retry logic,
idempotency persistence, LLM-based column mapping, and a full metrics
dashboard.

---

## Project Structure

```
smart_sync_scheduler/
├── app.py                        Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── logger.py                 File-based logger (graceful degradation)
│   ├── data_loader.py            CSV / XLSX / JSON unified loader
│   ├── schema_mapper.py          Rule-based column name mapping
│   ├── llm_mapper.py             LLM-based semantic mapping with caching
│   ├── validator.py              Data validation + timezone normalisation
│   ├── scheduler.py              FIFO / priority-queue scheduler
│   ├── conflict_resolver.py      Multi-level conflict resolution
│   ├── retry_handler.py          Retry logic + dead letter queue
│   ├── sync_engine.py            Atomic batch processing + idempotency
│   └── utils.py                  Config loading + DataFrame helpers
├── config/
│   ├── settings.json             All runtime configuration
│   └── schema_mapping.json       Known column name variations
├── data/
│   ├── sample.csv                Sample data with IST/UTC timestamps
│   ├── sample.xlsx               Sample data with non-standard column names
│   └── sample.json               Sample data with JSON schema variation
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Features

### Input Handling
Upload CSV, XLSX, or JSON files via the Streamlit UI. All three sample
files in `data/` demonstrate different column naming conventions.

### Schema Normalisation
Standard schema: `store_id`, `product`, `quantity`, `timestamp`, `update_id`

**Rule-based mapping**: `config/schema_mapping.json` lists known column
aliases for each standard field. Matching is case-insensitive.

**LLM-based mapping** (optional): When rule-based mapping cannot resolve all
required columns and `llm.enabled` is `true` in `settings.json`, the
application calls a configurable LLM endpoint (Ollama-compatible or
OpenAI-compatible). Results are cached on disk to avoid repeated calls.
Rate limiting prevents excessive API usage.

### Timezone Handling
All timestamps are normalised to UTC before processing. Timezone-aware
strings (e.g. `+05:30`, `+00:00`) are converted directly. Timezone-naive
strings are localised to `default_timezone` (configurable) then converted.
Unix epoch values (seconds or milliseconds) are also supported. Rows with
unparseable timestamps are dropped and logged.

### Scheduling
- **FIFO** (default): standard first-in, first-out using a deque.
- **Priority**: min-heap where stores with fewer processed records receive
  higher priority (prevents starvation).

Configure `scheduling_strategy` in `settings.json`.

### Conflict Resolution
Multi-level strategy applied within each batch:
1. Higher timestamp wins.
2. Equal timestamps: higher `update_id` (lexicographic) wins.
3. Equal update_ids: last record in batch wins (last-write-wins).

### Atomic Batch Processing
Updates are written to a temporary copy of the database. If all writes
succeed the copy becomes the live database. Any exception triggers a full
rollback to preserve consistency.

### Idempotency Persistence
Processed `update_id` values are saved to `data/processed_ids.json` after
every successful batch. Restarting the application does not allow duplicate
records to be processed again.

### Retry and Dead Letter Queue
Failed updates are retried up to `retry_limit` times (configurable). After
exhausting retries, the record is moved to an in-memory dead letter queue
visible in the UI.

### Metrics Dashboard
Per-cycle bar chart, work-completion donut, and queue-size-over-time line
chart. Summary metrics: processed, failed, retried, conflicts, duplicates.

---

## Configuration Reference

### settings.json

| Key                       | Default        | Description                                    |
|---------------------------|----------------|------------------------------------------------|
| max_sync_per_window       | 5              | Records pulled per sync window                 |
| conflict_policy           | multi_level    | latest_wins or multi_level                     |
| retry_limit               | 3              | Max retries before dead letter queue           |
| scheduling_strategy       | fifo           | fifo or priority                               |
| default_timezone          | UTC            | Assumed TZ for naive timestamps                |
| idempotency_store_path    | data/processed_ids.json | Persistent ID store path            |
| log_file                  | data/sync.log  | Log output path                                |
| llm.enabled               | false          | Enable LLM column mapping                     |
| llm.endpoint              | localhost:11434 | Ollama / OpenAI-compatible endpoint           |
| llm.model                 | qwen2.5:7b     | Model name to use                              |
| llm.confidence_threshold  | 0.7            | Minimum confidence to accept LLM mapping      |
| llm.rate_limit_per_minute | 10             | Maximum LLM API calls per 60 seconds          |

### schema_mapping.json

A dictionary where each key is a standard field name and each value is a
list of known column name aliases. Edit this file to add support for new
naming conventions without changing any code.

---

## Using the LLM Feature

1. Install and start [Ollama](https://ollama.com/): `ollama serve`
2. Pull a model: `ollama pull qwen2.5:7b`
3. In `config/settings.json`, set `"llm": {"enabled": true, ...}`
4. Restart the application.

OpenAI-compatible endpoints are also supported. Set `llm.endpoint` to
the appropriate URL and `llm.model` to the model name.

---

## Sample Files

| File           | Column names used                              |
|----------------|------------------------------------------------|
| sample.csv     | store_id, product, quantity, timestamp (standard) |
| sample.json    | branch, medicine, qty, recorded_at, id         |
| sample.xlsx    | shop_id, drug_name, stock, ts, version         |