# LLM-Driven Intent Extraction for UA Network Reconfiguration

Intent-based networking (IBN) framework for multi-UA urban delivery operations: natural
language coordination messages, classified by LLMs, drive radio access network
reconfiguration (bandwidth, transmit power, handover) in an O-RAN-inspired Mininet-WiFi
emulation.

## Repository structure

```
dataset/                        Synthetic dataset (CSV) and chat export (TXT) — published
prompts/                        Classification prompts used by the LLMs
data/                           Antenna/AP configuration and small test fixtures

scripts/
├── mininet_server_dynamic.py   Server: Mininet-WiFi topology, real RSSI, QoS, handover, DEFER
│
├── llm_reconfig.py             "Classify-then-map" client: LLM classifies, fixed table decides QoS
├── llm_policy_agent.py         "Policy agent" client: LLM decides bandwidth/power contextually
├── baseline_reactive.py        Reactive SSF baseline (no intent classification)
│
├── eval_real_chat.py           Evaluates models on the real chat (deterministic rules as ground truth)
├── eval_real_chat_deepseek.py  Same evaluation, DeepSeek-specific client
│
├── run_clean.py                Trusted orchestrator: restarts the server before each rep, validates
│                                and retries automatically (minimum drones, minimum accuracy, etc.)
├── run_rssi_batch.py           Runs the full campaign (all configurations, N reps each),
│                                keeping state in results/rssi_batch_state.json
│
├── plot_confusion_matrices.py  Generates confusion matrices (classify-then-map)
└── plot_network_impact.py      Generates packet loss and latency charts per configuration
```

`results/`, `figures/`, `paper/`, and `_chat_real.txt` (raw WhatsApp export, with participants'
personal data) are not versioned — see `.gitignore`. Only the synthetic dataset is published.

## Intent taxonomy

| Class | Description                      | Direction           |
|-------|-----------------------------------|----------------------|
| I0    | Neutral / administrative          | any                  |
| I1    | Takeoff request                   | pilot → controller   |
| I2    | Takeoff authorization              | controller → pilot   |
| I3    | In-flight report (position)        | pilot → controller   |
| I4    | Landing request                    | pilot → controller   |
| I5    | Landing authorization              | controller → pilot   |
| I6    | Ground confirmation (landed)       | pilot → controller   |
| I7    | Conflict alert                     | controller → any     |
| I8    | Mission pause (battery swap)       | pilot → controller   |

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Mininet-WiFi

Installed separately, outside pip — follow the official guide: https://mininet-wifi.github.io/

### 3. Environment variables

```bash
cp .env.example .env
# edit .env with your API keys and the VM's sudo password
export $(cat .env | xargs)   # or use python-dotenv / direnv
```

No key is hardcoded in the code — everything is read via `os.environ.get(...)`.

## How to run

All commands below assume you are at the repository root (data paths are relative to it,
not to the script).

### Single experiment (classify-then-map)

```bash
sudo python3 scripts/mininet_server_dynamic.py &   # starts the server (wait for "Ouvindo :5000")
python3 scripts/llm_reconfig.py --model qwen2.5:14b
```

Accepted models: `qwen2.5:14b`, `llama3.1:8b`, `ministral-8b-latest`, `deepseek-flash`,
plus the DeepInfra-served models (`meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, `Qwen/Qwen3-14B`).

Use `--classify-only` to run classification only, without network commands (fast, useful for
checking accuracy).

### Rule-based baseline (keyword + QoS table, no LLM)

Same pipeline as classify-then-map above — classification, bandwidth/power allocation via the
`INTENT_QOS` table, handover, conflict resolution (DEFER) — except classification is done by
deterministic rules (`classify_keyword()`) instead of an LLM. Answers the question "is an LLM
actually necessary, or do keyword rules already suffice?":

```bash
python3 scripts/llm_reconfig.py --model keyword-baseline
```

### Policy agent (contextual bandwidth/power)

```bash
python3 scripts/llm_policy_agent.py --model qwen2.5:14b
```

### Reactive baseline

```bash
python3 scripts/baseline_reactive.py --runs 1
```

### Full campaign (N reps per configuration, with validation and retry)

```bash
nohup python3 -u scripts/run_rssi_batch.py > /tmp/rssi_batch.log 2>&1 &
```

Restarts the Mininet-WiFi server before every repetition (guarantees a clean state — see the
`udp_sink.py` note in `run_clean.py`), validates each rep (minimum number of activated drones,
minimum classification accuracy) and saves incrementally to `results/rssi_batch_state.json`,
so it can be interrupted and resumed.

### Figures

```bash
python3 scripts/plot_confusion_matrices.py
python3 scripts/plot_network_impact.py
```

## Synthetic dataset

| Column      | Description                                              |
|-------------|------------------------------------------------------------|
| `timestamp` | Date and time of the message                               |
| `sender`    | Sender identifier (UAS-XX, UAV-XX, Control)                |
| `intent`    | Intent label (I0–I8)                                       |
| `mensagem`  | Message text                                                |
| `lat`, `lng`| GPS coordinates (Shanghai region, from LaDe)                |
| `ciclo_id`  | Mission cycle ID                                            |
| `fragmento` | Whether the message is a fragment of another                |

1,150 messages, combining real trajectories from the LaDe dataset (Shanghai) with the
linguistic patterns observed in real communications from the Interlagos Formula 1 Grand Prix.
