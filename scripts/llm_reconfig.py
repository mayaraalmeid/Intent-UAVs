#!/usr/bin/env python3
"""
LLM-driven network reconfiguration com classificação CONTEXTUAL.

Arquitetura:
  - Drones entram na simulação quando aparecem pela primeira vez no dataset
    (eles voam independentemente do LLM — a autorização não é necessária).
  - Ao aparecer, o drone recebe uma banda padrão (DEFAULT_BW = 500 kbps).
  - O LLM (non-RT) ajusta QoS conforme classifica as intenções.
  - Erros do LLM → banda errada → drones com QoS inadequado → degradação real.
  - O Near-RT RIC aplica os comandos via TCP (porta 5000).
"""
import argparse, csv, json, os, re, socket, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import anthropic, openai
from trajectory_interp import interpolate_trajectories, trajectory_stats

RIC_HOST        = 'localhost'
RIC_PORT        = 5000
TXT_FILE        = Path('dataset/uas_chat_final.txt')
CSV_FILE        = Path('dataset/uas_dataset_final.csv')
RESULTS_DIR     = Path('results')
MSG_DELAY       = 0.5
DEFAULT_BW  = 500   # kbps — banda padrão ao entrar na simulação (antes do LLM ajustar)

ANTHROPIC_KEY  = os.environ.get('ANTHROPIC_API_KEY', '')
OPENAI_KEY     = os.environ.get('OPENAI_API_KEY', '')
OLLAMA_KEY     = os.environ.get('OLLAMA_API_KEY', '')
OLLAMA_URL     = os.environ.get('OLLAMA_URL', 'https://cumbuco.ollama.atlab.ufc.br/ollama/v1')
MISTRAL_KEY    = os.environ.get('MISTRAL_API_KEY', '')
MISTRAL_URL    = 'https://api.mistral.ai/v1'
DEEPSEEK_KEY   = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL   = 'https://api.deepseek.com/v1'
GROQ_KEY       = os.environ.get('GROQ_API_KEY', '')
GROQ_URL       = 'https://api.groq.com/openai/v1'
DEEPINFRA_KEY  = os.environ.get('DEEPINFRA_API_KEY', '')
DEEPINFRA_URL  = 'https://api.deepinfra.com/v1/openai'

OLLAMA_MODELS = {
    'qwen2.5:14b': 90,
    'llama3.1:8b': 60,
    'qwen3:8b':    90,
    'phi4:latest': 90,
}
MISTRAL_MODELS  = {'ministral-8b-latest', 'mistral-small-latest', 'mistral-large-latest'}
DEEPSEEK_MODELS = {'deepseek-flash', 'deepseek-v4-pro'}
GROQ_MODELS     = {'qwen/qwen3.8-27b'}
DEEPINFRA_MODELS = {'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'Qwen/Qwen3-14B'}
KEYWORD_MODELS   = {'keyword-baseline'}

# Classificador baseline não-LLM: regras derivadas das pistas linguísticas da
# Taxonomy Table do SYSTEM_PROMPT acima.
def classify_keyword(sender, text, states, history):
    t0 = time.time()
    t = text.lower()
    ctrl = (sender == 'Control')
    intent = 'I0'
    if any(x in t for x in ['unauthorized drone', 'conflict zone', 'airspace violation', 'helicopter']):
        intent = 'I7'
    elif ctrl and any(x in t for x in ['cleared for takeoff', 'takeoff at your discretion', 'departure authorized']):
        intent = 'I2'
    elif ctrl and any(x in t for x in ['report when on ground', 'report on ground']):
        intent = 'I5'
    elif not ctrl and any(x in t for x in ['inbound for landing', 'on final approach', 'preparing to land', 'returning for landing']):
        intent = 'I4'
    elif not ctrl and any(x in t for x in ['requesting takeoff', 'awaiting takeoff authorization', 'requesting departure']):
        intent = 'I1'
    elif not ctrl and any(x in t for x in ['landed', 'on the ground']):
        intent = 'I6'
    elif not ctrl and any(x in t for x in ['battery swap', 'battery change']):
        intent = 'I8'
    elif not ctrl and re.search(r'\b(red|yellow|green|blue|orange|white)\s+zone\b', t) and re.search(r'\d+\s*m\b', t):
        intent = 'I3'

    if not ctrl:
        drone = sender
    else:
        m = re.search(r'\b([A-Za-z]+-\d+)\b', text)
        drone = m.group(1) if m else ''
    raw = json.dumps({'intent': intent, 'drone': drone})
    return raw, round((time.time() - t0) * 1000, 1)

# QoS por intent — aplicado quando LLM classifica corretamente
# I0 = padrão mínimo (drone presente mas LLM não identificou intent relevante)
INTENT_QOS = {
    'I0':    {'bw': DEFAULT_BW, 'tx': 'normal'},
    'I1':    {'bw': 500,        'tx': 'normal'},
    'I2':    {'bw': 2000,       'tx': 'increase'},
    'I3':    {'bw': 3000,       'tx': 'increase'},  # igual à demanda (Seção V-A, 3GPP) — sem folga
    'I4':    {'bw': 1000,       'tx': 'normal'},
    'I5':    {'bw': 500,        'tx': 'reduce'},
    'I6':    {'bw': DEFAULT_BW, 'tx': 'reduce'},
    # I7 = conflict alert FROM CONTROL — no direct network action on the sender.
    # The ConflictResolver handles the actual response: DEFER (100 kbps + tx=reduce)
    # is applied only to the queued drone. I7 itself is informational-only.
    'I7':    {'bw': DEFAULT_BW, 'tx': 'normal'},
    'I8':    {'bw': 300,        'tx': 'reduce'},
    # DEFER: drone queued due to airspace conflict — cell breathing to force handover
    'DEFER': {'bw': 100,        'tx': 'reduce'},
}

# Demanda real de tráfego por intent (ground truth) — independente da predição do LLM.
# O CBR sender ajusta a taxa para este valor quando a intent muda.
# Se alloc (INTENT_QOS[pred]) < demand → TC descarta pacotes → PDR real discrimina abordagens.
INTENT_DEMAND_KBPS = {
    'I0':    100,   # admin
    'I1':    500,   # takeoff: câmera + sensores
    'I2':    300,   # autorização
    'I3':   3000,   # em voo: vídeo HD + telemetria
    'I4':    800,   # landing prep
    'I5':    300,   # landing ack
    'I6':      0,   # pousado
    'I7':    400,   # conflito
    'I8':    100,   # bateria
    'DEFER': 100,   # aguardando
}

# Only trigger handover if drone moved more than this since last handover
HANDOVER_THRESHOLD_M = 50

# Mesmo prompt contextual do paper (janela 3 msgs + máquina de estados)
SYSTEM_PROMPT = """\
You are a network intent classifier for UAS (drone) systems in public safety operations.
Classify the CURRENT message using the taxonomy below.

## State Machine
Each UAS follows: pre_flight → airborne → landing → landed
You will be given the current state of relevant drones and the last 3 messages for context.
Use this to resolve ambiguous messages — especially from Control.

## Taxonomy

| Code | Name                        | Direction         | Key signals |
|------|-----------------------------|-------------------|-------------|
| I0   | No intent / administrative  | any               | system msgs, added to group; simple takeoff confirmations ("Airborne.", "Aircraft off ground.", "Off the ground.", "UAV-XX airborne."); simple standalone acks ("Acknowledged.", "Control copies.", "Copy 👍") |
| I1   | Takeoff request             | pilot → control   | "requesting takeoff", "awaiting takeoff authorization", "requesting departure", "pronto para decolar" |
| I2   | Takeoff authorization       | control → pilot   | "cleared for takeoff", "takeoff at your discretion", "departure authorized", "decolagem a critério" |
| I3   | In-flight position report   | pilot → control   | drone reports its CURRENT POSITION with zone and altitude while airborne: "UAV-44 — Yellow Zone, 30m, en route", "DRONE-22 in Yellow Zone at 50m", "RPA-83 reporting — Blue Zone, 30m" |
| I4   | Landing approach / prep     | pilot → control   | "inbound for landing", "on final approach", "preparing to land", "returning for landing", "regressando ao solo" |
| I5   | Landing acknowledgment      | control → pilot   | Control acknowledges a landing approach — context shows drone just reported landing prep (I4): "Copy, report when on ground", "Control copies, report on ground", "Acknowledged, report on ground" |
| I6   | Landed report               | pilot → control   | "aeronave em solo", "no solo", "drone em solo", "landed", "on the ground" |
| I7   | Conflict / safety alert     | control → any     | unauthorized drone detected, conflict zone, airspace violation, helicopter activity |
| I8   | Battery swap / standby      | pilot → control   | drone pausing for battery change: "battery swap, standby", "battery change — back in the air soon", "on ground for battery swap" |

## Critical rules

### I0 vs I3
- I0: Simple bare confirmation "Airborne." / "Aircraft off ground." / "UAV-XX off ground." immediately after takeoff clearance — no zone or altitude given.
- I3: Message includes BOTH a zone name AND an altitude (e.g. "Yellow Zone, 30m") while airborne.

### I0 vs I8
- I0: Short standalone acks with no operational content: "Acknowledged.", "Copy 👍", "Control copies.", "Copiado."
- I8: Explicit battery-swap notice from a pilot drone.

### I4 vs I5
- I4: The PILOT says they are approaching/preparing to land.
- I5: CONTROL responds to that with "Copy/report on ground". Use context: if the previous message was I4 from a drone, Control's "Copy..." → I5.

## Output
JSON only: {"intent": "IX", "drone": "callsign or empty string"}
"""

STATE_TRANSITIONS = {
    'I1': 'requesting_takeoff',
    'I2': 'cleared_takeoff',
    'I3': 'airborne',
    'I4': 'landing_approach',
    'I5': 'landing_approach',
    'I6': 'landed',
}


# ── Contexto / estado ─────────────────────────────────────────────────────────

def get_context_block(sender, text, states, history):
    lines = ['## Context']
    if history:
        lines.append('### Last messages in this channel:')
        for s, t, intent in history:
            lines.append(f'  [{intent}] {s}: {t[:80]}')
    mentioned = [d for d in states if d.lower() in text.lower() or d.lower() in sender.lower()]
    if not mentioned and states:
        active = {d: s for d, s in states.items()
                  if s in ('airborne', 'requesting_takeoff', 'cleared_takeoff', 'landing_approach')}
        if active:
            lines.append('### Active drones:')
            for d, s in list(active.items())[:5]:
                lines.append(f'  {d}: {s}')
    elif mentioned:
        lines.append('### Drone states:')
        for d in mentioned[:3]:
            lines.append(f'  {d}: {states.get(d, "unknown")}')
    lines.append(f'\n## Current message to classify\nSender: {sender}\nMessage: {text}')
    return '\n'.join(lines)

def update_state(states, drone, intent):
    if intent in STATE_TRANSITIONS and drone:
        states[drone] = STATE_TRANSITIONS[intent]


# ── Zone / altitude extraction ────────────────────────────────────────────────
_ZONE_RE = re.compile(r'\b(red|yellow|green|blue|orange|white)\s+zone\b', re.I)
_ALT_RE  = re.compile(r'\b(\d+)\s*(?:m\b|metros?\b|mts?\b|meters?\b)', re.I)

def extract_zone_altitude(text):
    zm = _ZONE_RE.search(text)
    am = _ALT_RE.search(text)
    return (zm.group(1).capitalize() if zm else None,
            int(am.group(1))         if am else 0)


def _latlon_dist_m(lat1, lon1, lat2, lon2):
    """Rough planar distance in metres (accurate enough for <50 km)."""
    from math import sqrt, cos, radians
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * 111_000 * cos(radians((lat1 + lat2) / 2))
    return sqrt(dlat**2 + dlon**2)


# ── Conflict Resolver (FCFS) ──────────────────────────────────────────────────
class ConflictResolver:
    """
    Detects simultaneous takeoff conflicts by zone + altitude separation.
    Deferred drones receive MINIMAL bandwidth + TX power reduction (cell breathing)
    to force handover to an alternative AP, physically separating them from the
    authorized drone. Released automatically when the conflicting drone lands (I6).
    """
    ALTITUDE_SEP = 90  # metres — minimum vertical separation

    def __init__(self):
        self._active: dict[str, dict] = {}  # drone → {zone, altitude, lat, lon}
        self._queue:  list[dict]      = []  # pending requests

    # ── Public interface ──────────────────────────────────────────────────────

    def request_takeoff(self, drone, zone, altitude, lat=0.0, lon=0.0):
        """Called when I1 classified. Returns list of (intent, target, lat, lon)."""
        if not zone:
            self._active[drone] = dict(zone=None, altitude=altitude, lat=lat, lon=lon)
            return [('I2', drone, lat, lon)]
        conflicts = self._find_conflicts(zone, altitude, exclude=drone)
        if not conflicts:
            self._active[drone] = dict(zone=zone, altitude=altitude, lat=lat, lon=lon)
            return [('I2', drone, lat, lon)]
        # Conflict: broadcast alert + defer new request (cell breathing)
        self._queue.append(dict(drone=drone, zone=zone, altitude=altitude, lat=lat, lon=lon))
        return [('I7',    'broadcast', 0.0,  0.0),
                ('DEFER', drone,       lat,  lon)]

    def landed(self, drone):
        """Called when I6 classified. Releases slot; returns any newly authorised actions."""
        self._active.pop(drone, None)
        actions, still_queued = [], []
        for req in self._queue:
            if not self._find_conflicts(req['zone'], req['altitude'], exclude=req['drone']):
                self._active[req['drone']] = req
                actions.append(('I2', req['drone'], req['lat'], req['lon']))
            else:
                still_queued.append(req)
        self._queue = still_queued
        return actions

    def check_airborne(self, drone, zone, altitude, lat=0.0, lon=0.0):
        """Called when I3 classified. Updates position; if trajectory now conflicts
        with an already-established drone in the same zone/altitude band, defer
        THIS drone (last to arrive at that position) via cell breathing — same
        treatment as a takeoff conflict, but without deactivating (already airborne).
        Self-releases: once a later check_airborne() finds no more conflict, DEFER
        stops being reissued and normal I3 QoS applies again on the next message."""
        if drone in self._active:
            self._active[drone].update(zone=zone, altitude=altitude, lat=lat, lon=lon)
        if zone and self._find_conflicts(zone, altitude, exclude=drone):
            return [('I7', 'broadcast', 0.0, 0.0), ('DEFER', drone, lat, lon)]
        return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_conflicts(self, zone, altitude, exclude=''):
        return [
            name for name, info in self._active.items()
            if name != exclude
            and info.get('zone') == zone
            and abs(info.get('altitude', 0) - altitude) < self.ALTITUDE_SEP
        ]


# ── Extração JSON ──────────────────────────────────────────────────────────────

def extract_intent(raw):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'^```[a-z]*\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            d = d[0]
        return d.get('intent', 'I0'), d.get('drone', '')
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"intent"\s*:\s*"(I\d)"[^{}]*\}', raw)
    if match:
        try:
            d = json.loads(match.group())
            return d.get('intent', 'I0'), d.get('drone', '')
        except Exception:
            pass
    m = re.search(r'I[0-8]', raw)
    return (m.group(0) if m else 'I0'), ''


# ── Classificadores ──────────────────────────────────────────────────────────

def classify_claude(sender, text, states, history):
    ctx = get_context_block(sender, text, states, history)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    t0 = time.time()
    msg = client.messages.create(
        model='claude-opus-4-7', max_tokens=128, system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': ctx}],
    )
    return msg.content[0].text.strip(), round((time.time()-t0)*1000, 1)

def classify_gpt(sender, text, states, history):
    ctx = get_context_block(sender, text, states, history)
    client = openai.OpenAI(api_key=OPENAI_KEY)
    for attempt in range(5):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model='gpt-4o', max_tokens=128,
                messages=[{'role':'system','content':SYSTEM_PROMPT},
                          {'role':'user','content':ctx}],
            )
            return resp.choices[0].message.content.strip(), round((time.time()-t0)*1000, 1)
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
    return '{"intent":"I0","drone":""}', 0

def classify_ollama(model, sender, text, states, history, timeout=90):
    ctx = get_context_block(sender, text, states, history)
    no_think = 'qwen3' in model
    prompt = ('/no_think\n' + SYSTEM_PROMPT) if no_think else SYSTEM_PROMPT
    client = openai.OpenAI(api_key=OLLAMA_KEY, base_url=OLLAMA_URL)

    def _call():
        resp = client.chat.completions.create(
            model=model, max_tokens=128, temperature=0, timeout=timeout,
            messages=[{'role':'system','content':prompt},
                      {'role':'user','content':ctx}],
        )
        return resp.choices[0].message.content.strip()

    for attempt in range(3):
        try:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                raw = future.result(timeout=timeout + 10)
            return raw, round((time.time()-t0)*1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout] tentativa {attempt+1}/3 — abortando chamada')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f'  [RateLimit] {wait}s (tentativa {attempt+1}/3)...')
                time.sleep(wait)
            else:
                print(f'  [Ollama ERROR] {err[:80]}')
                return '{"intent":"I0","drone":""}', 0
    return '{"intent":"I0","drone":""}', 0


def classify_mistral(model, sender, text, states, history, timeout=60):
    ctx = get_context_block(sender, text, states, history)
    client = openai.OpenAI(api_key=MISTRAL_KEY, base_url=MISTRAL_URL)

    def _call():
        resp = client.chat.completions.create(
            model=model, max_tokens=128, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': ctx}],
        )
        return resp.choices[0].message.content.strip()

    for attempt in range(3):
        try:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_call).result(timeout=timeout + 10)
            return raw, round((time.time() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout Mistral] tentativa {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f'  [RateLimit Mistral] {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [Mistral ERROR] {err[:80]}')
                return '{"intent":"I0","drone":""}', 0
    return '{"intent":"I0","drone":""}', 0


def classify_groq(model, sender, text, states, history, timeout=60):
    ctx = get_context_block(sender, text, states, history)
    client = openai.OpenAI(api_key=GROQ_KEY, base_url=GROQ_URL)

    def _call():
        resp = client.chat.completions.create(
            model=model, max_tokens=128, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': ctx}],
        )
        return resp.choices[0].message.content.strip()

    for attempt in range(3):
        try:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_call).result(timeout=timeout + 10)
            return raw, round((time.time() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout Groq] tentativa {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f'  [RateLimit Groq] {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [Groq ERROR] {err[:80]}')
                return '{"intent":"I0","drone":""}', 0
    return '{"intent":"I0","drone":""}', 0


def classify_deepinfra(model, sender, text, states, history, timeout=60):
    ctx = get_context_block(sender, text, states, history)
    client = openai.OpenAI(api_key=DEEPINFRA_KEY, base_url=DEEPINFRA_URL)
    # Qwen3 tem "thinking mode" ligado por padrão, que consome o max_tokens
    # inteiro "raciocinando" (fica em reasoning_content) sem nunca emitir o
    # JSON final em content. /no_think no system prompt desliga isso.
    sys_prompt = ('/no_think\n' + SYSTEM_PROMPT) if 'qwen3' in model.lower() else SYSTEM_PROMPT

    def _call():
        resp = client.chat.completions.create(
            model=model, max_tokens=128, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': sys_prompt},
                      {'role': 'user',   'content': ctx}],
        )
        return resp.choices[0].message.content.strip()

    for attempt in range(3):
        try:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_call).result(timeout=timeout + 10)
            return raw, round((time.time() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout DeepInfra] tentativa {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f'  [RateLimit DeepInfra] {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [DeepInfra ERROR] {err[:80]}')
                return '{"intent":"I0","drone":""}', 0
    return '{"intent":"I0","drone":""}', 0


def classify_deepseek(model, sender, text, states, history, timeout=60):
    ctx = get_context_block(sender, text, states, history)
    client = openai.OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL)

    def _call():
        resp = client.chat.completions.create(
            model=model, max_tokens=128, temperature=0, timeout=timeout,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user',   'content': ctx}],
            # thinking mode vem ligado por padrão no deepseek-flash e consome
            # todo o max_tokens "raciocinando" sem nunca emitir o JSON final.
            extra_body={'thinking': {'type': 'disabled'}},
        )
        return resp.choices[0].message.content.strip()

    for attempt in range(3):
        try:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_call).result(timeout=timeout + 10)
            return raw, round((time.time() - t0) * 1000, 1)
        except (FuturesTimeout, TimeoutError):
            print(f'  [Timeout DeepSeek] tentativa {attempt+1}/3')
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f'  [RateLimit DeepSeek] {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [DeepSeek ERROR] {err[:80]}')
                return '{"intent":"I0","drone":""}', 0
    return '{"intent":"I0","drone":""}', 0


# ── TCP ───────────────────────────────────────────────────────────────────────

def send_cmd(cmd):
    try:
        s = socket.socket(); s.settimeout(5)
        s.connect((RIC_HOST, RIC_PORT))
        s.sendall(json.dumps(cmd).encode() + b'\n')
        s.close()
        return True
    except Exception as e:
        print(f'  [TCP] {e}')
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='claude | gpt | qwen2.5:14b | llama3.1:8b | ...')
    parser.add_argument('--classify-only', action='store_true',
                        help='Só classifica, sem enviar comandos de rede (rápido)')
    args = parser.parse_args()

    model_id       = args.model
    classify_only  = args.classify_only
    ollama_timeout = OLLAMA_MODELS.get(model_id, 90)

    if model_id == 'claude':
        def classify(s, t, st, h): return classify_claude(s, t, st, h)
    elif model_id == 'gpt':
        def classify(s, t, st, h): return classify_gpt(s, t, st, h)
    elif model_id in MISTRAL_MODELS:
        def classify(s, t, st, h): return classify_mistral(model_id, s, t, st, h)
    elif model_id in DEEPSEEK_MODELS:
        def classify(s, t, st, h): return classify_deepseek(model_id, s, t, st, h)
    elif model_id in GROQ_MODELS:
        def classify(s, t, st, h): return classify_groq(model_id, s, t, st, h)
    elif model_id in DEEPINFRA_MODELS:
        def classify(s, t, st, h): return classify_deepinfra(model_id, s, t, st, h)
    elif model_id in KEYWORD_MODELS:
        def classify(s, t, st, h): return classify_keyword(s, t, st, h)
    else:
        def classify(s, t, st, h): return classify_ollama(model_id, s, t, st, h, ollama_timeout)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts_run = int(time.time())

    # Limpa estado do servidor entre runs
    send_cmd({'command': 'network_config', 'action': 'reset_all', 'drone': '', 'intent': ''})
    time.sleep(1)
    send_cmd({'command': 'network_config', 'action': 'reset_sink', 'drone': '', 'intent': ''})
    time.sleep(1)

    # ── Carrega dataset ───────────────────────────────────────────────────────
    txt_msgs = []
    txt_re = re.compile(r'^\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\] ([^:]+): (.+)$')
    with open(TXT_FILE, encoding='utf-8') as f:
        for line in f:
            m = txt_re.match(line.strip())
            if m:
                txt_msgs.append({'sender': m.group(2).strip(), 'text': m.group(3).strip()})

    csv_rows = []
    with open(CSV_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                csv_rows.append({
                    'sender':    row['sender'].strip(),
                    'gt_intent': row['intent'].strip(),
                    'lat':       float(row.get('lat') or 0),
                    'lon':       float(row.get('lng') or 0),
                })
            except Exception:
                pass

    messages = [{'sender': t['sender'], 'text': t['text'],
                 'gt_intent': g['gt_intent'], 'lat': g['lat'], 'lon': g['lon']}
                for t, g in zip(txt_msgs, csv_rows)]

    messages = interpolate_trajectories(messages)
    tstats = trajectory_stats(messages)
    print(f'Trajetórias interpoladas: {tstats["drones_with_trajectory"]} drones, '
          f'dist. média={tstats["avg_trajectory_length_m"]}m total={tstats["total_distance_km"]}km')

    mode_str = 'classify-only' if classify_only else 'contextual + ativação lazy'
    print(f'[{model_id}] {len(messages)} mensagens ({mode_str}). Iniciando...\n')

    # ── Reset do servidor ─────────────────────────────────────────────────────
    if not classify_only:
        print('[reset] Limpando estado anterior do servidor...')
        send_cmd({'command': 'network_config', 'action': 'reset_all'})
        time.sleep(1)

    activated        = set()
    deactivated      = set()
    conflict_resolver = ConflictResolver()
    last_positions   = {}   # drone → (lat, lon) — for handover threshold

    # ── Estado e contexto ─────────────────────────────────────────────────────
    states      = {}
    history     = deque(maxlen=3)
    predictions = []

    # ── Loop principal ────────────────────────────────────────────────────────
    for i, msg in enumerate(messages, 1):
        sender   = msg['sender']
        text     = msg['text'] or msg['gt_intent']
        lat, lon = msg['lat'], msg['lon']
        gt       = msg['gt_intent']
        gt_norm  = gt if gt.startswith('I') else 'I0'

        raw, api_ms = classify(sender, text, states, history)
        pred, drone = extract_intent(raw)

        update_state(states, drone, pred)
        if sender != 'Control':
            update_state(states, sender, pred)
        history.append((sender, text, pred))

        correct = (pred == gt_norm)
        predictions.append({'gt': gt_norm, 'pred': pred, 'correct': correct,
                             'latency_ms': api_ms, 'sender': sender,
                             'lat': lat, 'lon': lon})

        mark = '✓' if correct else '✗'
        print(f'  [{i:4}/{len(messages)}] {sender:15} gt={gt_norm:3} pred={pred:3} {mark} ({api_ms:.0f}ms)')

        if not classify_only:
            # ── Resolve conflitos e gera ações extras ─────────────────────────
            actor = drone or sender
            extra_actions = []   # (intent, target, lat, lon)

            if pred == 'I1' and sender != 'Control':
                zone, alt = extract_zone_altitude(text)
                extra_actions = conflict_resolver.request_takeoff(actor, zone, alt, lat, lon)

            elif pred == 'I6' and sender != 'Control':
                extra_actions = conflict_resolver.landed(actor)

            elif pred == 'I3' and sender != 'Control':
                zone, alt = extract_zone_altitude(text)
                if zone and alt:
                    extra_actions = conflict_resolver.check_airborne(actor, zone, alt, lat, lon)

            # ── Ativa drone na rede (lazy, na primeira mensagem) ─────────────
            if sender != 'Control':
                if sender not in activated and sender not in deactivated:
                    ok = send_cmd({'command': 'network_config', 'action': 'activate',
                                   'intent': 'I0', 'drone': sender, 'lat': lat, 'lon': lon,
                                   'bandwidth_kbps': DEFAULT_BW,
                                   'demand_kbps': INTENT_DEMAND_KBPS.get(gt, 100),
                                   'network_control': {'bandwidth_kbps': DEFAULT_BW, 'qos_class': 'BE'},
                                   'radio_access': {'tx_power': 'normal'}})
                    if ok:
                        activated.add(sender)
                        last_positions[sender] = (lat, lon)

            # ── Aplica QoS da mensagem classificada ───────────────────────────
            if sender != 'Control' and sender in activated:
                if pred == 'I6':
                    # Pouso confirmado: desativa drone
                    send_cmd({'command': 'network_config', 'action': 'deactivate',
                              'intent': pred, 'drone': sender})
                    activated.discard(sender)
                    deactivated.add(sender)

                elif pred == 'I1' and not extra_actions:
                    # I1 sem conflito — QoS normal; o I2 virá via extra_actions
                    pass

                elif pred not in ('I1',):
                    # Aplica QoS do intent classificado (alloc) + demanda real (ground truth).
                    # Envia a cada mensagem elegível — a deduplicação de tc qdisc replace
                    # redundante (mesma alocação de antes) agora é feita no servidor
                    # (set_link_params), que também garante que a demanda real (taxa do
                    # sender CBR) fique sempre atualizada mensagem a mensagem, mesmo
                    # quando a alocação de banda não muda.
                    qos = INTENT_QOS.get(pred, INTENT_QOS['I0'])
                    send_cmd({'command': 'network_config', 'action': 'bandwidth',
                              'drone': sender, 'intent': pred,
                              'bandwidth_kbps': qos['bw'],
                              'demand_kbps': INTENT_DEMAND_KBPS.get(gt, 100),
                              'radio_access': {'tx_power': qos['tx']}})

                    # Handover somente se drone moveu além do threshold
                    if lat or lon:
                        last = last_positions.get(sender)
                        dist = _latlon_dist_m(last[0], last[1], lat, lon) if last else HANDOVER_THRESHOLD_M + 1
                        if dist > HANDOVER_THRESHOLD_M:
                            send_cmd({'command': 'network_config', 'action': 'handover',
                                      'drone': sender, 'intent': pred, 'lat': lat, 'lon': lon,
                                      'demand_kbps': INTENT_DEMAND_KBPS.get(gt, 100)})
                            last_positions[sender] = (lat, lon)

            # ── Aplica ações do conflict resolver ────────────────────────────
            for act_intent, act_target, act_lat, act_lon in extra_actions:
                act_qos = INTENT_QOS.get(act_intent, INTENT_QOS['I0'])

                if act_intent == 'I7':
                    # Alerta de conflito: broadcast — sem reconfiguração de rede
                    print(f'  [CONFLICT] I7 broadcast: {act_target}')

                elif act_intent == 'DEFER':
                    # Drone em espera: banda mínima + reassociação forçada pra
                    # um AP diferente do atual (a potência do próprio drone não
                    # afeta o RSSI de downlink que ele mede do AP, então reduzir
                    # tx power sozinho não força handover nenhum — a separação
                    # física vem da reassociação explícita abaixo).
                    if act_target not in activated:
                        send_cmd({'command': 'network_config', 'action': 'activate',
                                  'intent': 'DEFER', 'drone': act_target,
                                  'lat': act_lat, 'lon': act_lon,
                                  'bandwidth_kbps': act_qos['bw'],
                                  'demand_kbps': INTENT_DEMAND_KBPS['DEFER'],
                                  'network_control': {'bandwidth_kbps': act_qos['bw'], 'qos_class': 'BE'},
                                  'radio_access': {'tx_power': act_qos['tx']}})
                        activated.add(act_target)
                        last_positions[act_target] = (act_lat, act_lon)
                    else:
                        send_cmd({'command': 'network_config', 'action': 'bandwidth',
                                  'drone': act_target, 'intent': 'DEFER',
                                  'bandwidth_kbps': act_qos['bw'],
                                  'demand_kbps': INTENT_DEMAND_KBPS['DEFER'],
                                  'radio_access': {'tx_power': act_qos['tx']}})
                    send_cmd({'command': 'network_config', 'action': 'handover',
                              'drone': act_target, 'intent': 'DEFER',
                              'lat': act_lat, 'lon': act_lon, 'force': True})
                    print(f'  [DEFER] {act_target} queued — forced reassociation')

                elif act_intent == 'I2':
                    # Drone autorizado após fila liberada
                    i2_qos = INTENT_QOS['I2']
                    send_cmd({'command': 'network_config', 'action': 'bandwidth',
                              'drone': act_target, 'intent': 'I2',
                              'bandwidth_kbps': i2_qos['bw'],
                              'demand_kbps': INTENT_DEMAND_KBPS['I2'],
                              'radio_access': {'tx_power': i2_qos['tx']}})
                    print(f'  [QUEUE RELEASE] {act_target} authorized from queue')

            time.sleep(MSG_DELAY)

    # ── Salva métricas ────────────────────────────────────────────────────────
    if not classify_only:
        print('\nSalvando métricas...')
        send_cmd({'command': 'network_config', 'action': 'save_metrics', 'drone': '', 'intent': ''})
    time.sleep(3)

    total   = len(predictions)
    correct = sum(1 for p in predictions if p['correct'])
    acc     = round(correct / total * 100, 2) if total else 0
    avg_lat = round(sum(p['latency_ms'] for p in predictions) / total, 1) if total else 0

    safe_model = model_id.replace(':', '_').replace('/', '_')
    net_files = sorted(RESULTS_DIR.glob('network_metrics_*.json'))
    if net_files:
        net_data = json.load(open(net_files[-1]))
        net_data['summary']['mode']  = f'{model_id}_reconfig_contextual'
        net_data['summary']['model'] = model_id
        net_data['summary']['llm_accuracy_pct'] = acc
        out = RESULTS_DIR / f'{safe_model}_network_metrics_{ts_run}.json'
        json.dump(net_data, open(out, 'w'), indent=2)
        s = net_data['summary']
        print(f'\n=== REDE ===')
        print(f"  drones={s['total_drones_activated']}  handovers={s['total_handovers']}  "
              f"latency={s['avg_latency_ms']}ms  loss={s['avg_packet_loss_pct']}%  "
              f"conn_losses={s['total_connection_losses']}")
        print(f'  → {out}')

    llm_out = RESULTS_DIR / f'{safe_model}_llm_metrics_{ts_run}.json'
    json.dump({'mode': f'{model_id}_reconfig_contextual', 'model': model_id,
               'total_messages': total, 'accuracy_pct': acc,
               'avg_api_latency_ms': avg_lat, 'predictions': predictions},
              open(llm_out, 'w'), indent=2)
    print(f'\n=== LLM ===')
    print(f'  accuracy={acc}%  avg_api_latency={avg_lat}ms')
    print(f'  → {llm_out}')

if __name__ == '__main__':
    main()
