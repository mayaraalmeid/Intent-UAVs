#!/usr/bin/env python3
"""
Classificador contextual com máquina de estados por drone.

Melhoria sobre o baseline:
  - Estado por drone: pre_flight → airborne → landing → landed
  - Janela de contexto: últimas 3 mensagens da conversa
  - Prompt enriquecido com estado atual do(s) drone(s) mencionado(s)

Foco: aumentar recall de I5 (Landing Ack) e I2 (Takeoff Auth).
"""
import re, csv, json, os, time, sys, argparse
from pathlib import Path
from collections import defaultdict, deque

import anthropic

TXT_FILE    = Path('uas_chat_final (2).txt')
CSV_FILE    = Path('uas_dataset_final (1).csv')
RESULTS_DIR = Path('/home/wifi/datasets/results')

SYSTEM_PROMPT_BASE = """\
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

# ── State machine ─────────────────────────────────────────────────────────────
STATES = {}   # drone_callsign → state string
HISTORY = deque(maxlen=3)  # últimas 3 mensagens globais [(sender, text, intent)]

TRANSITIONS = {
    'I1': 'requesting_takeoff',
    'I2': 'cleared_takeoff',
    'I3': 'airborne',
    'I4': 'landing_approach',
    'I5': 'landing_approach',
    'I6': 'landed',
}

def get_context_block(sender, text):
    """Monta o bloco de contexto a injetar no prompt."""
    lines = ["## Context"]

    # Histórico recente
    if HISTORY:
        lines.append("### Last messages in this channel:")
        for s, t, intent in HISTORY:
            lines.append(f"  [{intent}] {s}: {t[:80]}")

    # Estado dos drones mencionados na mensagem atual
    mentioned = [d for d in STATES if d.lower() in text.lower()
                 or d.lower() in sender.lower()]
    if not mentioned and STATES:
        # Injecta estado de todos os drones ativos (airborne/requesting)
        active = {d: s for d, s in STATES.items()
                  if s in ('airborne', 'requesting_takeoff', 'cleared_takeoff', 'landing')}
        if active:
            lines.append("### Active drones:")
            for d, s in list(active.items())[:5]:
                lines.append(f"  {d}: {s}")
    else:
        if mentioned:
            lines.append("### Drone states:")
            for d in mentioned[:3]:
                lines.append(f"  {d}: {STATES.get(d, 'unknown')}")

    lines.append(f"\n## Current message to classify\nSender: {sender}\nMessage: {text}")
    return "\n".join(lines)

def update_state(drone, intent):
    if intent in TRANSITIONS and drone:
        STATES[drone] = TRANSITIONS[intent]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    txt_re = re.compile(r'^\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\] ([^:]+): (.+)', re.M)
    txt_msgs = {(m.group(2).strip(), m.group(3).strip()): m.group(3).strip()
                for m in txt_re.finditer(open(TXT_FILE, encoding='utf-8').read())}

    messages = []
    for row in csv.DictReader(open(CSV_FILE, encoding='utf-8')):
        sender = row.get('sender', '').strip()
        gt     = row.get('intent', '').strip()
        gt     = gt if gt.startswith('I') else 'I0'
        text   = row.get('mensagem', row.get('message', row.get('text', ''))).strip()
        if not text:
            key = next(((s, t) for s, t in txt_msgs if s == sender), None)
            text = txt_msgs.get(key, '') if key else ''
        try:
            lat = float(row.get('lat', 0) or 0)
            lon = float(row.get('lng', row.get('lon', 0)) or 0)
        except:
            lat = lon = 0.0
        messages.append({'sender': sender, 'text': text, 'gt': gt, 'lat': lat, 'lon': lon})
    return messages

# ── Classifier ───────────────────────────────────────────────────────────────
def classify(client, sender, text, model='claude'):
    context_msg = get_context_block(sender, text)
    t0 = time.time()
    if model == 'gpt':
        import openai
        gpt_client = openai.OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY',''),
            timeout=30.0,
        )
        raw = '{"intent": "I0", "drone": ""}'
        for attempt in range(6):
            try:
                resp = gpt_client.chat.completions.create(
                    model='gpt-4o',
                    max_tokens=120,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT_BASE},
                        {'role': 'user',   'content': context_msg},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                break
            except openai.RateLimitError as e:
                wait = min(2 ** attempt, 30)
                print(f'    [RateLimit] attempt {attempt+1}/6, sleeping {wait}s...', flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f'    [GPT error] {type(e).__name__}: {e}', flush=True)
                break
    else:
        resp = client.messages.create(
            model='claude-opus-4-7',
            max_tokens=120,
            system=SYSTEM_PROMPT_BASE,
            messages=[{'role': 'user', 'content': context_msg}],
        )
        raw = resp.content[0].text.strip()
    ms = round((time.time() - t0) * 1000, 1)
    # Strip markdown code fences if present
    raw = re.sub(r'^```[a-z]*\s*', '', raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        d      = json.loads(raw)
        intent = d.get('intent', 'I0')
        drone  = d.get('drone', '')
    except Exception:
        intent, drone = 'I0', ''
    return intent, drone, ms

# ── Metrics ───────────────────────────────────────────────────────────────────
def recall_report(predictions):
    from collections import defaultdict, Counter
    tp = defaultdict(int); fn = defaultdict(int); fp = defaultdict(int)
    for p in predictions:
        gt, pred = p['gt'], p['pred']
        if gt == pred: tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pred] += 1
    print(f"\n{'Intent':<8} {'TP':>4} {'FN':>4} {'Recall':>8}  {'GT total':>8}")
    for i in sorted(set(list(tp) + list(fn))):
        total = tp[i] + fn[i]
        rec   = tp[i] / total if total else 0.0
        print(f"  {i:<6} {tp[i]:>4} {fn[i]:>4} {rec:>8.2%}  {total:>8}")
    total_tp = sum(tp.values())
    total    = total_tp + sum(fn.values())
    print(f"\n  Accuracy: {total_tp}/{total} = {total_tp/total*100:.2f}%")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='claude', choices=['claude', 'gpt'])
    args = parser.parse_args()

    key = os.environ.get('ANTHROPIC_API_KEY', '')
    client = anthropic.Anthropic(api_key=key) if key else None

    messages = load_data()
    print(f'[Contextual-{args.model}] {len(messages)} mensagens. Iniciando...\n')

    predictions = []
    for i, msg in enumerate(messages, 1):
        sender, text, gt = msg['sender'], msg['text'], msg['gt']
        if not text:
            predictions.append({'gt': gt, 'pred': 'I0', 'sender': sender})
            HISTORY.append((sender, '(empty)', 'I0'))
            continue

        intent, drone, ms = classify(client, sender, text, model=args.model)

        # Atualiza estado do drone identificado
        update_state(drone, intent)
        # Também atualiza pelo sender se for um drone
        if sender != 'Control':
            update_state(sender, intent)

        # Histórico global
        HISTORY.append((sender, text, intent))

        gt_norm = gt if gt.startswith('I') else 'I0'
        correct = (intent == gt_norm)
        predictions.append({'gt': gt_norm, 'pred': intent, 'correct': correct,
                            'sender': sender, 'drone': drone, 'api_ms': ms})

        mark = '✓' if correct else '✗'
        print(f'  [{i:4}/{len(messages)}] {sender:15} gt={gt_norm} pred={intent} {mark} ({ms:.0f}ms)')

    recall_report(predictions)

    # Salva
    ts  = int(time.time())
    out = RESULTS_DIR / f'{args.model}_contextual_llm_{ts}.json'
    total   = len(predictions)
    correct = sum(1 for p in predictions if p.get('correct'))
    acc     = round(correct / total * 100, 2)
    json.dump({'mode': 'claude_contextual', 'accuracy_pct': acc,
               'total_messages': total, 'predictions': predictions},
              open(out, 'w'), indent=2)
    print(f'\n→ {out}')

if __name__ == '__main__':
    main()
