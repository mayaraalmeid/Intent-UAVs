#!/usr/bin/env python3

import json, os, time, sys
from pathlib import Path
from collections import deque
import openai

from eval_real_chat import (
    CHAT_FILE, RESULTS_DIR, SYSTEM_PROMPT,
    parse_chat, assign_gt, context_block, update_state,
    extract_json, compute_metrics,
)

DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1'
MODEL = 'deepseek-flash'


def classify_deepseek(client, sender, text, states, history, timeout=60):
    ctx = context_block(sender, text, states, history)
    t0 = time.time()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=512, temperature=0, timeout=timeout,
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                          {'role': 'user',   'content': ctx}],
                extra_body={'thinking': {'type': 'disabled'}},
            )
            raw = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = min(30 * (attempt + 1), 120)
                print(f'  [rate-limit] wait {wait}s...')
                time.sleep(wait)
            else:
                return 'I0', '', round((time.time() - t0) * 1000, 1), err
    else:
        return 'I0', '', round((time.time() - t0) * 1000, 1), 'max retries'

    ms = round((time.time() - t0) * 1000, 1)
    d = extract_json(raw)
    if d:
        return d.get('intent', 'I0'), d.get('drone', ''), ms, None
    return 'I0', '', ms, f'parse error: {raw[:60]}'


def main():
    print('Parsing real WhatsApp chat...')
    raw_msgs = parse_chat(CHAT_FILE)
    msgs_gt = assign_gt(raw_msgs)
    print(f'Total messages: {len(msgs_gt)}')

    client = openai.OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL)

    states = {}
    history = deque(maxlen=3)
    preds = []
    errors = 0

    for i, m in enumerate(msgs_gt):
        pred, drone, ms, err = classify_deepseek(client, m['sender'], m['text'], states, list(history))
        if err:
            errors += 1
            if errors <= 5:
                print(f'  [{i:3d}] ERROR: {err[:80]}')
        update_state(states, m['text'], pred)
        history.append((m['sender'], m['text'], pred))
        preds.append({'idx': i, 'sender': m['sender'], 'text': m['text'][:80],
                       'gt': m['gt'], 'pred': pred, 'ms': ms})
        if (i + 1) % 50 == 0:
            corr = sum(1 for p in preds if p['gt'] == p['pred'])
            print(f'  [{i+1:3d}/{len(msgs_gt)}] acc={corr/len(preds)*100:.1f}%  errors={errors}')

    acc, macro_f1, per_class = compute_metrics(preds)
    print(f'\n--- {MODEL} ---')
    print(f'  Accuracy : {acc*100:.2f}%')
    print(f'  Macro-F1 : {macro_f1*100:.2f}%')
    print(f'  Errors   : {errors}')
    for l, v in sorted(per_class.items()):
        print(f"    {l}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}  n={v['support']}")

    result = {'model': MODEL, 'n_messages': len(msgs_gt),
              'accuracy_pct': round(acc * 100, 2), 'macro_f1_pct': round(macro_f1 * 100, 2),
              'errors': errors, 'per_class': per_class, 'predictions': preds}
    out = RESULTS_DIR / 'real_eval_deepseek_flash.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
