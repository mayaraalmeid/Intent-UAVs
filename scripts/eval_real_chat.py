#!/usr/bin/env python3


import re, json, os, time, sys
from pathlib import Path
from collections import defaultdict, deque
import openai

OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://cumbuco.ollama.atlab.ufc.br/ollama/v1")
CHAT_FILE  = Path("_chat_real.txt")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_MODELS = ["qwen2.5:14b", "llama3.1:8b"]

# ── Same system prompt as paper ───────────────────────────────────────────────
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
| I0   | No intent / administrative  | any               | system msgs, added to group; bare callsigns with no action ("CAvPM-1", "RPA GCM 38", "F1 Drone - Up", "UAS CPAM-10"); simple airborne confirmations ("Airborne.", "fora de solo", "aeronave fora de solo", "Reportando fora do solo", "Aircraft off ground."); simple standalone acks ("Acknowledged.", "Copy 👍", "ciente", "roger", "copiado", "positivo", "tks", "Lagos ciente", "Coordenação ciente") |
| I1   | Takeoff request             | pilot → control   | "requesting takeoff", "awaiting takeoff authorization", "pronto para decolar", "solicitação para decolagem", "aguarda autorização para decolagem", "aguardando autorização" |
| I2   | Takeoff authorization       | control → pilot   | "cleared for takeoff", "takeoff at your discretion", "decolagem a critério", "decolagem autorizada"; IMPORTANT: "decolagem a critério na área verde, altura 40m" is still I2 even though it mentions a zone and altitude — the zone/altitude are constraints on the takeoff clearance, not a position report |
| I3   | In-flight position report   | pilot → control   | drone reports BOTH a zone name AND an altitude in the same message while airborne: "UAV-44 — Yellow Zone, 30m, en route", "área verde, 40m em rota", "zona amarela, 30m"; a message with ONLY altitude ("30m", "30 metros") or ONLY zone ("área amarela") is I0, not I3 |
| I4   | Landing approach / prep     | pilot → control   | "inbound for landing", "preparing to land", "preparando para pouso", "solicitação para pouso", "pronto para pouso", "realizando pouso", "regressando ao solo" |
| I5   | Landing acknowledgment      | control → pilot   | Control acknowledges a landing approach — context shows drone just reported landing prep (I4): "Copy, report when on ground", "ciente, reporte no solo", "Lagos ciente, reporte no solo", "Autorizado, reporte no solo", "Coordenação ciente, reporte no solo" |
| I6   | Landed report               | pilot → control   | pilot reports that the drone HAS LANDED: "aeronave em solo", "no solo", "drone em solo", "solo", "landed", "F1 Drone - Landed"; IMPORTANT: "fora do solo" / "aeronave fora de solo" / "Reportando fora do solo" = airborne (off the ground) = I0, NOT I6 |
| I7   | Conflict / safety alert     | control → any     | unauthorized drone detected, conflict zone, airspace violation, "drone detectado", "mavic sem contato", "drone avistado no perímetro" |
| I8   | Battery swap / standby      | pilot → control   | drone pausing for battery change: "battery swap, standby", "troca de bateria", "on ground for battery swap" |

## Critical rules

### I0 vs I3
- I0: Simple confirmation "Airborne." / "fora de solo" / "aeronave fora de solo" — no zone AND no altitude given. A message with ONLY altitude ("30m", "30 metros", "30Metros") or ONLY zone ("área amarela") → I0.
- I3: Message includes BOTH a zone name AND an altitude in the SAME message (e.g. "Yellow Zone, 30m", "área verde, 40m").

### I0 vs I6 — direction of movement matters
- I6: Drone is now ON THE GROUND — "em solo", "no solo", "ao solo", "landed", "solo".
- I0: Drone just LEFT the ground — "fora do solo", "fora de solo", "aeronave fora de solo", "Reportando fora do solo", "off ground". These mean AIRBORNE, not landed.
- Also I0: bare callsign check-ins with no verb ("CAvPM-1", "RPA GCM 38"). These are acknowledgment pings, not landing reports.

### I5 vs I6 — "reporte no solo" direction
- I5 (control → pilot): Control INSTRUCTS the pilot to report landing: "reporte no solo", "report on the ground", "report when on ground". The controller is asking for a future report.
- I6 (pilot → control): Pilot CONFIRMS they are on the ground: "aeronave em solo", "no solo", "solo", "landed". The pilot is reporting current status.
- KEY: "reporte fora do solo" / "reporte fora de solo" = control asking pilot to report when airborne = I0 (not I5, not I6).

### I2 vs I3 — takeoff clearance with constraints
- I2: Control grants takeoff with optional zone/altitude constraints: "decolagem a critério na área verde, altura 40m" = I2 (the zone and altitude describe WHERE/HOW HIGH to fly, not a current position).
- I3: Pilot reports current in-flight position. Only pilots send I3.

### I0 vs I8
- I0: Short standalone acks with no operational content: "ciente", "Copiado.", "Lagos ciente", "Coordenação ciente" (when not following an I4).
- I8: Explicit battery-swap notice from a pilot drone: "troca de bateria", "battery swap".

### I4 vs I5
- I4: The PILOT says they are approaching/preparing to land.
- I5: CONTROL responds — either with "ciente"/"copy" + "reporte no solo"/"report on ground", OR with a standalone "reporte no solo" instruction. Any time control says "reporte no solo" it is I5.

## Output
JSON only: {"intent": "IX", "drone": "callsign or empty string"}

Messages may be in Portuguese or English — classify regardless of language.
"""

# ── Parse real WhatsApp chat ──────────────────────────────────────────────────
MSG_RE = re.compile(r'^\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)', re.MULTILINE)
SYSTEM_MSG = re.compile(
    r'adicionou|criou este grupo|removeu|Mensagem apagada|<anexado:|Localização:|'
    r'As mensagens e ligações são protegidas|criptografia', re.I)

# Mapeamento explícito baseado em análise das mensagens do chat real
# Identificação anonimizada: os nomes/apelidos reais dos participantes do GP de
# Interlagos foram removidos por privacidade (não publicados em lugar nenhum,
# nem no artigo nem aqui). Para rodar contra o chat real original (privado,
# não incluso neste repositório), preencha CONTROLLERS/PILOTS com os
# identificadores reais localmente — nunca comite esse mapeamento.
CONTROLLERS = {
    'Controller-01', 'Controller-02', 'Controller-03', 'Controller-04',
    'Controller-05', 'Controller-06', 'Controller-07', 'Controller-08',
    'Controller-09', 'Controller-10', 'Controller-11',
}

PILOTS = {
    'Pilot-01', 'Pilot-02', 'Pilot-03', 'Pilot-04', 'Pilot-05', 'Pilot-06',
    'Pilot-07', 'Pilot-08', 'Pilot-09', 'Pilot-10', 'Pilot-11', 'Pilot-12',
    'Pilot-13', 'Pilot-14', 'Pilot-15', 'Pilot-16',
}

def is_control(sender):
    return sender in CONTROLLERS


def parse_chat(path):
    text = path.read_text(encoding='utf-8')
    text = text.replace('\u200E','').replace('\u202A','').replace('\u202C','')
    rows = []
    for date, time_, sender, body in MSG_RE.findall(text):
        sender = sender.strip().lstrip('~').strip()
        body   = body.strip()
        if SYSTEM_MSG.search(body):
            continue
        rows.append({'date': date, 'time': time_, 'sender': sender, 'text': body})
    return rows




# ── Rule-based ground truth ───────────────────────────────────────────────────
def gt_label(sender, text, prev_labels):
    t    = re.sub(r'\s+', ' ', text.lower()).strip()
    ctrl = is_control(sender)

    # I7 — conflict/safety alert
    if any(x in t for x in ['drone detectado','mavic','sem contato com a torre',
                              'drone avistado','protocolo de abordagem',
                              'desencadeando','area proibida','área proibida']):
        # Filter out "mavic" that's not about conflict (e.g. brand mentions)
        if any(x in t for x in ['detectado','sem contato','avistado','protocolo',
                                  'desencadeando','proibida']):
            return 'I7'
        if 'mavic' in t and 'sem contato' in t:
            return 'I7'

    # I2 — takeoff authorization (control → pilot)
    if ctrl and any(x in t for x in ['decolagem a critério','decolagem à critério',
                                       'takeoff at your discretion','cleared for takeoff',
                                       'decolagem autorizada']):
        return 'I2'

    # I5 — landing acknowledgment (control → pilot)
    # "reporte no solo"/"report on ground" from control is ALWAYS I5 — it is the specific
    # landing instruction; there is no other operational meaning for this phrase.
    if ctrl and any(x in t for x in ['reporte no solo','report when on ground',
                                      'report on ground','report on the ground']):
        return 'I5'
    last_op = next((l for l in reversed(prev_labels) if l not in ('I0', None)), None)
    if ctrl and last_op == 'I4':
        if any(x in t for x in ['autorizado','ciente','lagos ciente',
                                  'coordenação lagos ciente','coordenacao lagos ciente',
                                  'coordenação ciente']):
            return 'I5'

    # I4 — landing approach/prep (pilot)
    if any(x in t for x in ['preparando para pouso','pronto para pouso',
                              'realizando pouso','inbound for landing','on final approach',
                              'f1 drone - landing']):
        return 'I4'
    if 'solicitação para pouso' in t and 'reporte' not in t:
        return 'I4'
    if 'para pouso' in t and not ctrl and 'reporte' not in t and 'ciente' not in t:
        return 'I4'

    # I1 — takeoff request (pilot)
    if any(x in t for x in ['solicitação para decolagem','solicitando autorização',
                              'aguarda autorização','aguardando autorização',
                              'pronto para decolagem','pronto pra decolagem',
                              'permission to take off','requesting takeoff',
                              'requesting departure','permissão para decolagem',
                              'permissao para decolagem','solicitando permissão para',
                              'solicitando permissao para']):
        return 'I1'
    # "Para ciência equipe... aguarda autorização para decolagem"
    if 'autorização' in t and 'decolagem' in t and not ctrl:
        return 'I1'
    if 'autorizacao' in t and 'decolagem' in t and not ctrl:
        return 'I1'

    # I6 — relato de pouso do PILOTO. Nunca "reporte no solo" (instrução do controle).
    if any(x in t for x in ['aeronave em solo','drone em solo','uas em solo',
                              'aeronave em sono','f1 drone - landed','f1 drone - land']):
        return 'I6'
    if not ctrl:
        if re.search(r'\bem\s+solo\b|\bao\s+solo\b', t):
            return 'I6'
        if re.search(r'\bno\s+solo\b', t) and 'reporte' not in t:
            return 'I6'
        if re.fullmatch(r'solo[\s.!]*', t.strip()):
            return 'I6'
        if re.search(r'(uas|rpa|uav|cavpm|cpam|gcm|cavam|aeronave)\S*\s+(em|ao|no)\s+solo', t):
            return 'I6'

    # I8 — battery swap standby (pilot, explicitly about battery while on ground)
    if 'troca de bateria' in t and not any(x in t for x in ['decolagem','pouso','voo']):
        return 'I8'

    # I3 — in-flight position report (zone + altitude)
    zone_pat = re.search(r'(área|area|turno|turn|zona)\s+\w+', t)
    alt_pat  = re.search(r'\d+\s*(?:m\b|metros?\b|mts?\b|f[te]\b|feet\b)', t)
    if zone_pat and alt_pat and not ctrl:
        if not any(x in t for x in ['decolagem','autorização','autorizacao','critério','criterio']):
            return 'I3'

    return 'I0'


def assign_gt(messages):
    prev_labels = deque(maxlen=5)
    result = []
    for m in messages:
        label = gt_label(m['sender'], m['text'], list(prev_labels))
        result.append({**m, 'gt': label})
        prev_labels.append(label)
    return result


# ── Context block (same methodology as paper) ─────────────────────────────────
def context_block(sender, text, states, history):
    lines = ["## Context"]
    if history:
        lines.append("### Last messages in this channel:")
        for s, t, intent in history:
            lines.append(f"  [{intent}] {s}: {t[:80]}")
    active = {d: s for d, s in states.items()
              if s in ("airborne","requesting_takeoff","cleared_takeoff","landing_approach")}
    if active:
        lines.append("### Active drones:")
        for d, s in list(active.items())[:5]:
            lines.append(f"  {d}: {s}")
    lines.append(f"\n## Current message to classify\nSender: {sender}\nMessage: {text}")
    return "\n".join(lines)


TRANSITIONS = {
    "I1": "requesting_takeoff",
    "I2": "cleared_takeoff",
    "I3": "airborne",
    "I4": "landing_approach",
    "I5": "landing_approach",
    "I6": "landed",
}


def update_state(states, text, intent):
    m = re.search(r'\b(UAS|RPA|UAV|DRONE|CAVPM|CPAM|GCM|CAvPM|F1 Drone)\S*', text, re.I)
    drone = m.group(0).strip() if m else ""
    if intent in TRANSITIONS and drone:
        states[drone] = TRANSITIONS[intent]
    return drone


# ── JSON extractor ────────────────────────────────────────────────────────────
def extract_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r'\{[^{}]*"intent"\s*:\s*"I\d+"[^{}]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    for m in re.finditer(r'\{[^{}]+\}', raw):
        try:
            return json.loads(m.group())
        except Exception:
            continue
    return None


# ── Classify ──────────────────────────────────────────────────────────────────
def classify(client, sender, text, states, history, model, timeout=90):
    ctx = context_block(sender, text, states, history)
    t0  = time.time()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": ctx},
                ],
                temperature=0,
                timeout=timeout,
            )
            raw = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(30*(attempt+1), 120)
                print(f"  [rate-limit] wait {wait}s...")
                time.sleep(wait)
            else:
                return "I0", "", round((time.time()-t0)*1000, 1), err
    else:
        return "I0", "", round((time.time()-t0)*1000, 1), "max retries"

    ms = round((time.time()-t0)*1000, 1)
    d  = extract_json(raw)
    if d:
        return d.get("intent","I0"), d.get("drone",""), ms, None
    return "I0", "", ms, f"parse error: {raw[:60]}"


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for p in preds:
        gt, pred = p["gt"], p["pred"]
        if gt == pred:
            tp[gt] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gt]   += 1
    labels = sorted(set(p["gt"] for p in preds) | set(p["pred"] for p in preds))
    per_class = {}
    for l in labels:
        prec = tp[l]/(tp[l]+fp[l]) if (tp[l]+fp[l]) else 0.0
        rec  = tp[l]/(tp[l]+fn[l]) if (tp[l]+fn[l]) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        per_class[l] = {"precision": round(prec,4), "recall": round(rec,4),
                         "f1": round(f1,4), "support": tp[l]+fn[l]}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    acc      = correct / len(preds) if preds else 0.0
    return acc, macro_f1, per_class


# ── Run one model ─────────────────────────────────────────────────────────────
def run_model(client, model, messages_gt, timeout=90):
    print(f"\n{'='*60}\nModel: {model}  ({len(messages_gt)} msgs)\n{'='*60}")
    states  = {}
    history = deque(maxlen=3)
    preds   = []
    errors  = 0

    for i, m in enumerate(messages_gt):
        pred, drone, ms, err = classify(client, m["sender"], m["text"],
                                         states, list(history), model, timeout)
        if err:
            errors += 1
            if errors <= 5:
                print(f"  [{i:3d}] ERROR: {err[:80]}")
        update_state(states, m["text"], pred)
        history.append((m["sender"], m["text"], pred))
        preds.append({"idx": i, "sender": m["sender"], "text": m["text"][:80],
                       "gt": m["gt"], "pred": pred, "ms": ms})
        if (i+1) % 50 == 0:
            corr = sum(1 for p in preds if p["gt"]==p["pred"])
            print(f"  [{i+1:3d}/{len(messages_gt)}] acc={corr/len(preds)*100:.1f}%  errors={errors}")

    acc, macro_f1, per_class = compute_metrics(preds)
    print(f"\n--- {model} ---")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Macro-F1 : {macro_f1*100:.2f}%")
    print(f"  Errors   : {errors}")
    for l, v in sorted(per_class.items()):
        print(f"    {l}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}  n={v['support']}")

    return {"model": model, "n_messages": len(messages_gt),
            "accuracy_pct": round(acc*100,2), "macro_f1_pct": round(macro_f1*100,2),
            "errors": errors, "per_class": per_class, "predictions": preds}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    print("Parsing real WhatsApp chat...")
    raw_msgs = parse_chat(CHAT_FILE)
    msgs_gt  = assign_gt(raw_msgs)

    from collections import Counter
    dist = Counter(m["gt"] for m in msgs_gt)
    print(f"Total messages: {len(msgs_gt)}")
    for k in sorted(dist): print(f"  {k}: {dist[k]}")

    gt_path = RESULTS_DIR / "real_chat_gt.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump([{"idx": i, "sender": m["sender"], "text": m["text"], "gt": m["gt"]}
                   for i,m in enumerate(msgs_gt)], f, ensure_ascii=False, indent=2)
    print(f"GT saved: {gt_path}")

    client = openai.OpenAI(api_key=OLLAMA_KEY, base_url=OLLAMA_URL)

    all_results = []
    for model in models:
        result = run_model(client, model, msgs_gt)
        all_results.append(result)
        slug = model.replace(':','_').replace('.','_').replace('-','_')
        out = RESULTS_DIR / f"real_eval_{slug}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out}")

    print("\n" + "="*65)
    print(f"{'Model':35s} {'Acc%':>8} {'MacroF1%':>10} {'Errors':>8}")
    print("-"*65)
    for r in all_results:
        print(f"{r['model']:35s} {r['accuracy_pct']:>8.2f} {r['macro_f1_pct']:>10.2f} {r['errors']:>8}")


if __name__ == "__main__":
    main()
