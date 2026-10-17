#!/usr/bin/env python3

import argparse, csv, json, os, re, socket, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
import statistics

import openai
from trajectory_interp import interpolate_trajectories, trajectory_stats

# ── Config ────────────────────────────────────────────────────────────────────
RIC_HOST    = 'localhost'
RIC_PORT    = 5000
TXT_FILE    = Path('dataset/uas_chat_final.txt')
CSV_FILE    = Path('dataset/uas_dataset_final.csv')
RESULTS_DIR = Path('results')
MSG_DELAY   = 0.5   # igual ao llm_reconfig.py — garante comparação justa de loss/latência de rede

OLLAMA_KEY  = os.environ.get('OLLAMA_API_KEY', '')
OLLAMA_URL  = os.environ.get('OLLAMA_URL', 'https://cumbuco.ollama.atlab.ufc.br/ollama/v1')
OLLAMA_TIMEOUTS = {'qwen2.5:14b': 90, 'llama3.1:8b': 60}

MISTRAL_KEY = os.environ.get('MISTRAL_API_KEY', '')
MISTRAL_URL = 'https://api.mistral.ai/v1'
MISTRAL_TIMEOUTS = {'ministral-8b-latest': 60, 'mistral-small-latest': 60,
                    'mistral-large-latest': 90}

DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1'
DEEPSEEK_TIMEOUTS = {'deepseek-flash': 60, 'deepseek-v4-pro': 90}

OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
GPT_TIMEOUTS = {'gpt-4o': 60}

DEEPINFRA_KEY = os.environ.get('DEEPINFRA_API_KEY', '')
DEEPINFRA_URL = 'https://api.deepinfra.com/v1/openai'
DEEPINFRA_TIMEOUTS = {'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': 60,
                      'Qwen/Qwen3-14B': 60}

# Demanda real (ground truth) por intenção — mesma tabela do llm_reconfig.py.
# Sem isso o servidor nunca atualiza o CBR sender além do baseline inicial de
# 50kbps: a rede nunca é estressada de verdade e loss/latência ficam
# artificialmente baixos independente da banda alocada.
INTENT_DEMAND_KBPS = {
    'I0': 100, 'I1': 500, 'I2': 300, 'I3': 3000, 'I4': 800,
    'I5': 300, 'I6': 0, 'I7': 400, 'I8': 100, 'DEFER': 100,
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Non-RT RIC (Radio Intelligence Controller) for a UAV coordination system.
For each incoming coordination message, you receive the current network state and
must output BOTH the intent classification AND the optimal network configuration.

The policy is NOT a fixed table: adapt bandwidth and TX power to the actual
network state (zone congestion, drone history, resource availability).

## Intent taxonomy

| Code | Description | Direction |
|------|-------------|-----------|
| I0 | Administrative / no operational intent | any |
| I1 | Takeoff request | pilot → control |
| I2 | Takeoff authorization | control → pilot |
| I3 | In-flight position report (requires BOTH zone AND altitude) | pilot → control |
| I4 | Landing approach / preparation | pilot → control |
| I5 | Landing acknowledgment by Control | control → pilot |
| I6 | Landed confirmation | pilot → control |
| I7 | Conflict / safety alert | control → any |
| I8 | Battery swap / standby | pilot → control |

## Network action vocabulary

**action** (pick exactly one):
- "activate"   — first message from this drone: open connection with given BW/TX
- "bandwidth"  — adjust BW + TX for already-active drone
- "handover"   — trigger AP reassignment (use when zone/altitude changed)
- "deactivate" — drone landed; close session
- "defer"      — conflict detected; hold drone with minimal BW + TX reduce (cell breathing)
- "none"       — Control message or pure administrative: no network change

**bandwidth_kbps** (integer, 100–4000):
- I1 (on ground, waiting): 300–500
- I2 (cleared, about to fly): 1500–2000
- I3 (active flight): 2000–4000
- I4 (approaching): 800–1500
- I6 / deactivate: omit (not relevant)
- I8 (battery swap): 200–300
- DEFER (conflict queue): always 100
- ⚠ REDUCE by ~25% if the zone has ≥3 active drones (congested)
- ⚠ INCREASE up to max if message mentions priority or emergency

**tx_power** ("increase" | "normal" | "reduce"):
- "increase": takeoff, active flight — extends AP coverage toward drone
- "normal":   standby, ground ops
- "reduce":   deferred / landed — shrinks AP cell (cell breathing), forcing
              deferred drone at cell edge to hand over to neighbour AP,
              achieving physical airspace separation without commands

**handover** (true | false):
- true: drone position likely changed (different zone or altitude than last report)
- false: same position, just BW adjustment

## Decision rules

1. I6 → action="deactivate" always.
2. Sender="Control" → action="none" (control messages don't carry pilot network state).
3. Two drones in same zone with altitude difference <90m → second one gets action="defer".
4. I3 → set handover=true only if zone or altitude differs from drone's last known position.
5. I7 → action="none" (ConflictResolver handles DEFER separately; just classify the alert).
6. I0 → action="none".

## Output

Return ONLY valid JSON, no extra text or markdown:
{"intent":"IX","drone":"callsign or empty","action":"...","bandwidth_kbps":N,"tx_power":"...","handover":true,"rationale":"one concise sentence"}
"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(sender, text, net_state, history):
    """
    Assembles the user message: network state snapshot + recent messages + new message.
    The LLM uses this to make context-aware policy decisions.
    """
    lines = ["## Current network state"]

    active = net_state.get('active', {})
    if active:
        lines.append(f"Active drones ({len(active)}):")
        for d, info in list(active.items())[:12]:
            zone = info.get('zone') or 'unknown zone'
            alt  = info.get('altitude', 0)
            bw   = info.get('bw_kbps', 500)
            lines.append(f"  {d}: {zone}, {alt}m, {bw} kbps")
    else:
        lines.append("No active drones.")

    deferred = net_state.get('deferred', [])
    if deferred:
        lines.append(f"Deferred/queued (conflict): {', '.join(deferred)}")

    # Zone congestion summary
    zone_counts: dict[str, int] = {}
    for info in active.values():
        z = info.get('zone')
        if z:
            zone_counts[z] = zone_counts.get(z, 0) + 1
    congested = [f"{z}({n} drones)" for z, n in zone_counts.items() if n >= 3]
    if congested:
        lines.append(f"⚠ Congested zones: {', '.join(congested)}")

    lines.append("\n## Recent messages (with classified intent)")
    if history:
        for s, t, intent in history:
            lines.append(f"  [{intent}] {s}: {t[:90]}")
    else:
        lines.append("  (none yet)")

    lines.append(f"\n## New message\nSender: {sender}\nMessage: {text}")
    return '\n'.join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

_ZONE_RE = re.compile(r'\b(red|yellow|green|blue|orange|white)\s+zone\b', re.I)
_ALT_RE  = re.compile(r'\b(\d+)\s*(?:m\b|metros?\b|mts?\b|meters?\b)', re.I)


def extract_zone_alt(text):
    zm = _ZONE_RE.search(text)
    am = _ALT_RE.search(text)
    return (zm.group(1).capitalize() if zm else None,
            int(am.group(1)) if am else 0)


# Faixas de bandwidth_kbps por intenção — mesmas informadas no SYSTEM_PROMPT.
# A LLM às vezes ignora a orientação numérica (ex: I3 saindo com 150kbps, fora
# da faixa 2000-4000); aplicamos clamp aqui para garantir que a decisão final
# respeite a faixa, preservando a variação contextual dentro dela.
BW_RANGES = {
    'I1': (300, 500),
    'I2': (1500, 2000),
    'I3': (2000, 4000),
    'I4': (800, 1500),
    'I8': (200, 300),
}


def clamp_bandwidth(intent: str, action: str, bw: int) -> int:
    if action == 'defer':
        return 100
    lo_hi = BW_RANGES.get(intent)
    if lo_hi is None:
        return bw  # I0/I5/I6/I7: sem faixa definida no prompt, não relevante
    lo, hi = lo_hi
    return max(lo, min(hi, bw))


def parse_policy(raw: str) -> dict:
    """Extract policy dict from LLM output, with fallback."""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'^```[a-z]*\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw).strip()
    # Alguns modelos (ex: llama3.1:8b) ignoram "responda só JSON" e escrevem
    # um preâmbulo em markdown antes do bloco JSON. Extrai o ÚLTIMO bloco
    # {...} de qualquer lugar do texto em vez de assumir que raw é só JSON.
    blocks = re.findall(r'\{[^{}]*\}', raw, flags=re.DOTALL)
    candidate = blocks[-1] if blocks else raw
    try:
        d = json.loads(candidate)
        if isinstance(d, list):
            d = d[0]
        intent = d.get('intent', 'I0')
        action = d.get('action', 'none')
        bw     = d.get('bandwidth_kbps', 500)
        bw     = int(bw) if bw else 500
        bw     = clamp_bandwidth(intent, action, bw)
        return {
            'intent':         intent,
            'drone':          d.get('drone', ''),
            'action':         action,
            'bandwidth_kbps': bw,
            'tx_power':       d.get('tx_power') or 'normal',
            'handover':       bool(d.get('handover', False)),
            'rationale':      d.get('rationale', ''),
        }
    except Exception:
        pass
    # Fallback: extract at least the intent code
    m = re.search(r'I[0-8]', raw)
    return {
        'intent': m.group(0) if m else 'I0',
        'drone': '', 'action': 'none',
        'bandwidth_kbps': 500, 'tx_power': 'normal',
        'handover': False, 'rationale': '',
    }


def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * p / 100) - 1)
    return round(s[idx], 1)


# ── Ollama API call (with latency measurement) ────────────────────────────────

def call_ollama(model: str, user_msg: str, timeout: int = 90):
    """Returns (raw_text, infer_ms). Retries up to 3× on timeout/rate-limit."""
    client = openai.OpenAI(api_key=OLLAMA_KEY, base_url=OLLAMA_URL)
    sys = ('/no_think\n' + SYSTEM_PROMPT) if 'qwen3' in model else SYSTEM_PROMPT

    def _request():
        r = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': sys},
                      {'role': 'user',   'content': user_msg}],
        )
        return r.choices[0].message.content.strip()

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_request).result(timeout=timeout + 10)
            return raw, round((time.perf_counter() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout] attempt {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30 * (attempt + 1), 120)
                print(f'  [RateLimit] waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [API ERROR] {err[:100]}')
                return '{"intent":"I0","action":"none"}', 0.0
    return '{"intent":"I0","action":"none"}', 0.0


def call_mistral(model: str, user_msg: str, timeout: int = 60):
    """Returns (raw_text, infer_ms). Mesma lógica de retry do call_ollama."""
    client = openai.OpenAI(api_key=MISTRAL_KEY, base_url=MISTRAL_URL)

    def _request():
        r = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': user_msg}],
        )
        return r.choices[0].message.content.strip()

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_request).result(timeout=timeout + 10)
            return raw, round((time.perf_counter() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout Mistral] attempt {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30 * (attempt + 1), 120)
                print(f'  [RateLimit] waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [API ERROR] {err[:100]}')
                return '{"intent":"I0","action":"none"}', 0.0
    return '{"intent":"I0","action":"none"}', 0.0


def call_deepseek(model: str, user_msg: str, timeout: int = 60):
    """Returns (raw_text, infer_ms). Mesma lógica de retry do call_ollama/call_mistral."""
    client = openai.OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL)

    def _request():
        r = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': user_msg}],
            # thinking mode vem ligado por padrão e consome todo o max_tokens
            # "raciocinando" sem nunca emitir o JSON final.
            extra_body={'thinking': {'type': 'disabled'}},
        )
        return r.choices[0].message.content.strip()

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_request).result(timeout=timeout + 10)
            return raw, round((time.perf_counter() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout DeepSeek] attempt {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30 * (attempt + 1), 120)
                print(f'  [RateLimit] waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [API ERROR] {err[:100]}')
                return '{"intent":"I0","action":"none"}', 0.0
    return '{"intent":"I0","action":"none"}', 0.0


def call_gpt(model: str, user_msg: str, timeout: int = 60):
    """Returns (raw_text, infer_ms). Mesma lógica de retry das demais."""
    client = openai.OpenAI(api_key=OPENAI_KEY)

    def _request():
        r = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': user_msg}],
        )
        return r.choices[0].message.content.strip()

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_request).result(timeout=timeout + 10)
            return raw, round((time.perf_counter() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout GPT] attempt {attempt+1}/3')
        except openai.RateLimitError:
            wait = min(30 * (attempt + 1), 120)
            print(f'  [RateLimit] waiting {wait}s...')
            time.sleep(wait)
        except Exception as e:
            print(f'  [API ERROR] {str(e)[:100]}')
            return '{"intent":"I0","action":"none"}', 0.0
    return '{"intent":"I0","action":"none"}', 0.0


def call_deepinfra(model: str, user_msg: str, timeout: int = 60):
    """Returns (raw_text, infer_ms). Mesma lógica de retry das demais."""
    client = openai.OpenAI(api_key=DEEPINFRA_KEY, base_url=DEEPINFRA_URL)
    # Qwen3 tem "thinking mode" ligado por padrão (consome o max_tokens todo
    # raciocinando, sem emitir o JSON final) — /no_think desliga isso.
    sys_prompt = ('/no_think\n' + SYSTEM_PROMPT) if 'qwen3' in model.lower() else SYSTEM_PROMPT

    def _request():
        r = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': sys_prompt},
                      {'role': 'user',   'content': user_msg}],
        )
        return r.choices[0].message.content.strip()

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_request).result(timeout=timeout + 10)
            return raw, round((time.perf_counter() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout DeepInfra] attempt {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30 * (attempt + 1), 120)
                print(f'  [RateLimit] waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [API ERROR] {err[:100]}')
                return '{"intent":"I0","action":"none"}', 0.0
    return '{"intent":"I0","action":"none"}', 0.0


def call_llm(model: str, user_msg: str, timeout: int):
    """Dispatcher: roteia pro provedor certo conforme o nome do modelo."""
    if model in MISTRAL_TIMEOUTS:
        return call_mistral(model, user_msg, timeout)
    if model in DEEPSEEK_TIMEOUTS:
        return call_deepseek(model, user_msg, timeout)
    if model in GPT_TIMEOUTS:
        return call_gpt(model, user_msg, timeout)
    if model in DEEPINFRA_TIMEOUTS:
        return call_deepinfra(model, user_msg, timeout)
    return call_ollama(model, user_msg, timeout)


# ── Mininet TCP command (with RTT measurement) ────────────────────────────────

def send_cmd(cmd: dict):
    """Returns (success, rtt_ms)."""
    t0 = time.perf_counter()
    try:
        s = socket.socket()
        s.settimeout(5)
        s.connect((RIC_HOST, RIC_PORT))
        s.sendall(json.dumps(cmd).encode() + b'\n')
        s.close()
        return True, round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        print(f'  [TCP] {e}')
        return False, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LLM Policy Agent — contextual network config with latency measurement')
    parser.add_argument('--model', default='qwen2.5:14b',
                        choices=list(OLLAMA_TIMEOUTS.keys()) + list(MISTRAL_TIMEOUTS.keys())
                                + list(DEEPSEEK_TIMEOUTS.keys()) + list(GPT_TIMEOUTS.keys())
                                + list(DEEPINFRA_TIMEOUTS.keys()),
                        help='Ollama, Mistral, DeepSeek, GPT ou DeepInfra model to use')
    parser.add_argument('--classify-only', action='store_true',
                        help='Skip network commands (latency benchmark only)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only the first N messages (0 = all)')
    args = parser.parse_args()

    model    = args.model
    cls_only = args.classify_only
    limit    = args.limit
    timeout  = (OLLAMA_TIMEOUTS.get(model) or MISTRAL_TIMEOUTS.get(model)
                or DEEPSEEK_TIMEOUTS.get(model) or GPT_TIMEOUTS.get(model)
                or DEEPINFRA_TIMEOUTS.get(model, 90))
    RESULTS_DIR.mkdir(exist_ok=True)
    ts_run = int(time.time())

    # ── Load dataset ──────────────────────────────────────────────────────────
    txt_re = re.compile(r'^\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\] ([^:]+): (.+)$')
    txt_msgs: list[dict] = []
    with open(TXT_FILE, encoding='utf-8') as f:
        for line in f:
            m = txt_re.match(line.strip())
            if m:
                txt_msgs.append({'sender': m.group(2).strip(), 'text': m.group(3).strip()})

    csv_rows: list[dict] = []
    with open(CSV_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                csv_rows.append({
                    'gt':  row['intent'].strip(),
                    'lat': float(row.get('lat') or 0),
                    'lon': float(row.get('lng') or 0),
                })
            except Exception:
                pass

    messages = [
        {'sender': t['sender'], 'text': t['text'],
         'gt': g['gt'] if g['gt'].startswith('I') else 'I0',
         'lat': g['lat'], 'lon': g['lon']}
        for t, g in zip(txt_msgs, csv_rows)
    ]
    if limit:
        messages = messages[:limit]

    messages = interpolate_trajectories(messages)
    tstats = trajectory_stats(messages)
    print(f'Trajetórias: {tstats["drones_with_trajectory"]} drones, '
          f'dist. média={tstats["avg_trajectory_length_m"]}m '
          f'total={tstats["total_distance_km"]}km')

    mode = 'classify-only' if cls_only else 'policy-agent + Mininet'
    print(f'[{model}] {len(messages)} messages | mode: {mode}')
    print(f'Measuring: infer_ms + proc_ms + cmd_ms = e2e_ms\n')

    if not cls_only:
        send_cmd({'command': 'network_config', 'action': 'reset_all', 'drone': '', 'intent': ''})
        time.sleep(1)
        send_cmd({'command': 'network_config', 'action': 'reset_sink', 'drone': '', 'intent': ''})
        time.sleep(1)

    # ── State ─────────────────────────────────────────────────────────────────
    history   = deque(maxlen=3)
    net_state = {
        'active':   {},   # drone → {zone, altitude, bw_kbps, lat, lon}
        'deferred': [],
    }
    activated: set[str] = set()
    records:   list[dict] = []

    # ── Message loop ──────────────────────────────────────────────────────────
    for i, msg in enumerate(messages, 1):
        sender   = msg['sender']
        text     = msg['text']
        lat, lon = msg['lat'], msg['lon']
        gt       = msg['gt']

        # Build context for LLM
        ctx = build_context(sender, text, net_state, history)

        # ── LLM inference ────────────────────────────────────────────────────
        t_llm_start = time.perf_counter()
        raw, infer_ms = call_llm(model, ctx, timeout)
        t_after_llm = time.perf_counter()

        policy = parse_policy(raw)
        intent = policy['intent']
        actor  = policy['drone'] or sender

        proc_ms = round((time.perf_counter() - t_after_llm) * 1000, 1)

        # ── Network command ──────────────────────────────────────────────────
        cmd_ms  = 0.0
        applied = False

        # Control messages never carry a pilot network node; guard regardless of LLM output
        if sender == 'Control':
            policy['action'] = 'none'

        if not cls_only and sender != 'Control' and policy['action'] != 'none':
            action = policy['action']
            bw     = policy['bandwidth_kbps']
            tx     = policy['tx_power']

            if action == 'activate' or (actor not in activated and intent not in ('I0', 'I5')):
                ok, cmd_ms = send_cmd({
                    'command': 'network_config', 'action': 'activate',
                    'intent': intent, 'drone': actor, 'lat': lat, 'lon': lon,
                    'bandwidth_kbps': bw,
                    'demand_kbps': INTENT_DEMAND_KBPS.get(gt, 100),
                    'network_control': {'bandwidth_kbps': bw, 'qos_class': 'BE'},
                    'radio_access': {'tx_power': tx},
                })
                if ok:
                    activated.add(actor)
                    applied = True

            elif action == 'deactivate' or intent == 'I6':
                ok, cmd_ms = send_cmd({
                    'command': 'network_config', 'action': 'deactivate',
                    'intent': intent, 'drone': actor,
                })
                activated.discard(actor)
                net_state['active'].pop(actor, None)
                applied = ok

            elif action == 'defer':
                # Banda mínima + reassociação forçada pra um AP diferente do
                # atual — reduzir só a tx power do drone não afeta o RSSI de
                # downlink que ele mede (depende só da potência do AP), então
                # a separação física precisa vir de uma reassociação explícita.
                ok, cmd_ms = send_cmd({
                    'command': 'network_config', 'action': 'bandwidth',
                    'drone': actor, 'intent': 'DEFER',
                    'bandwidth_kbps': 100,
                    'demand_kbps': INTENT_DEMAND_KBPS['DEFER'],
                    'radio_access': {'tx_power': 'reduce'},
                })
                send_cmd({
                    'command': 'network_config', 'action': 'handover',
                    'drone': actor, 'intent': 'DEFER',
                    'lat': lat, 'lon': lon, 'force': True,
                })
                if ok and actor not in net_state['deferred']:
                    net_state['deferred'].append(actor)
                applied = ok

            else:
                # bandwidth or handover
                ok, bw_ms = send_cmd({
                    'command': 'network_config', 'action': 'bandwidth',
                    'drone': actor, 'intent': intent,
                    'bandwidth_kbps': bw,
                    'demand_kbps': INTENT_DEMAND_KBPS.get(gt, 100),
                    'radio_access': {'tx_power': tx},
                })
                cmd_ms = bw_ms
                applied = ok
                if policy['handover'] and (lat or lon):
                    _, ho_ms = send_cmd({
                        'command': 'network_config', 'action': 'handover',
                        'drone': actor, 'intent': intent, 'lat': lat, 'lon': lon,
                    })
                    cmd_ms += ho_ms

            # Update local network state snapshot
            if intent not in ('I6', 'I0') and actor and sender != 'Control':
                zone, alt = extract_zone_alt(text)
                net_state['active'].setdefault(actor, {}).update(
                    zone=zone, altitude=alt, bw_kbps=bw, lat=lat, lon=lon)
                if actor in net_state['deferred'] and action != 'defer':
                    net_state['deferred'].remove(actor)

        e2e_ms  = round((time.perf_counter() - t_llm_start) * 1000, 1)
        correct = (intent == gt)
        history.append((sender, text[:90], intent))

        records.append({
            'idx':           i,
            'sender':        sender,
            'gt':            gt,
            'pred':          intent,
            'correct':       correct,
            'action':        policy['action'],
            'bandwidth_kbps': policy['bandwidth_kbps'],
            'tx_power':      policy['tx_power'],
            'handover':      policy['handover'],
            'rationale':     policy['rationale'],
            'infer_ms':      infer_ms,
            'proc_ms':       proc_ms,
            'cmd_ms':        cmd_ms,
            'e2e_ms':        e2e_ms,
            'applied':       applied,
        })

        mark = '✓' if correct else '✗'
        print(f'  [{i:4}/{len(messages)}] {sender[:15]:15} gt={gt} pred={intent} {mark} '
              f'| infer={infer_ms:5.0f}ms e2e={e2e_ms:5.0f}ms '
              f'| {policy["action"]:10} {policy["bandwidth_kbps"]:4}kbps {policy["tx_power"]}')

        time.sleep(MSG_DELAY)

    # ── Save metrics and finalize ──────────────────────────────────────────────
    if not cls_only:
        send_cmd({'command': 'network_config', 'action': 'save_metrics', 'drone': '', 'intent': ''})
        time.sleep(3)

    # ── Statistics ────────────────────────────────────────────────────────────
    total    = len(records)
    n_correct = sum(1 for r in records if r['correct'])
    acc      = round(n_correct / total * 100, 2) if total else 0

    infers = [r['infer_ms'] for r in records if r['infer_ms'] > 0]
    e2es   = [r['e2e_ms']   for r in records if r['e2e_ms'] > 0]
    cmds   = [r['cmd_ms']   for r in records if r['cmd_ms'] > 0]

    lat_stats = {
        'inference_ms': {
            'mean':  round(statistics.mean(infers), 1)  if infers else 0,
            'stdev': round(statistics.stdev(infers), 1) if len(infers) > 1 else 0,
            'p25':   percentile(infers, 25),
            'p50':   percentile(infers, 50),
            'p75':   percentile(infers, 75),
            'p95':   percentile(infers, 95),
            'p99':   percentile(infers, 99),
            'n':     len(infers),
        },
        'cmd_rtt_ms': {
            'mean': round(statistics.mean(cmds), 1) if cmds else 0,
            'p50':  percentile(cmds, 50),
            'p95':  percentile(cmds, 95),
        },
        'e2e_ms': {
            'mean': round(statistics.mean(e2es), 1) if e2es else 0,
            'p50':  percentile(e2es, 50),
            'p95':  percentile(e2es, 95),
            'p99':  percentile(e2es, 99),
        },
    }

    safe_model = model.replace(':', '_').replace('/', '_')

    # Attach to network metrics file
    if not cls_only:
        net_files = sorted(RESULTS_DIR.glob('network_metrics_*.json'))
        if net_files:
            net_data = json.load(open(net_files[-1]))
            net_data['summary']['mode']             = f'policy_agent_{model}'
            net_data['summary']['model']            = model
            net_data['summary']['llm_accuracy_pct'] = acc
            net_data['summary']['api_latency_stats'] = lat_stats
            out_net = RESULTS_DIR / f'policy_{safe_model}_network_{ts_run}.json'
            json.dump(net_data, open(out_net, 'w'), indent=2)
            s = net_data['summary']
            print(f'\n=== NETWORK ===')
            print(f"  handovers={s['total_handovers']}  "
                  f"latency={s['avg_latency_ms']}ms  "
                  f"loss={s['avg_packet_loss_pct']}%  "
                  f"conn_losses={s['total_connection_losses']}")
            print(f'  → {out_net}')

    out_llm = RESULTS_DIR / f'policy_{safe_model}_llm_{ts_run}.json'
    json.dump({
        'mode':           f'policy_agent_{model}',
        'model':          model,
        'total_messages': total,
        'accuracy_pct':   acc,
        'api_latency_stats': lat_stats,
        'records':        records,
    }, open(out_llm, 'w'), indent=2)

    print(f'\n=== LLM POLICY AGENT — {model} ===')
    print(f'  accuracy    : {acc}%  ({n_correct}/{total} correct)')
    i_s = lat_stats['inference_ms']
    e_s = lat_stats['e2e_ms']
    print(f'  infer (ms)  : mean={i_s["mean"]}  p50={i_s["p50"]}  '
          f'p95={i_s["p95"]}  p99={i_s["p99"]}  σ={i_s["stdev"]}')
    print(f'  E2E   (ms)  : mean={e_s["mean"]}  p50={e_s["p50"]}  '
          f'p95={e_s["p95"]}  p99={e_s["p99"]}')
    if cmds:
        c_s = lat_stats['cmd_rtt_ms']
        print(f'  cmd RTT(ms) : mean={c_s["mean"]}  p50={c_s["p50"]}  p95={c_s["p95"]}')
    print(f'  → {out_llm}')


if __name__ == '__main__':
    main()
