#!/usr/bin/env python3

import json, os, statistics
import run_clean as rc

RESULTS_DIR = rc.RESULTS_DIR
STATE_JSON  = os.path.join(RESULTS_DIR, 'rssi_batch_state.json')
APPROACH_BY_NAME = {a['name']: a for a in rc.APPROACHES}

N_REPS = 30  # mesmo N usado nos experimentos originais (validade estatistica)

PLAN = [
    ('baseline_reactive',          N_REPS),
    ('keyword-baseline',           N_REPS),
    ('ministral-8b-latest',        N_REPS),
    ('ministral-8b-latest-policy', N_REPS),
    ('llama3.1:8b',                N_REPS),
    ('llama3.1:8b-policy',         N_REPS),
    ('deepseek-flash',             N_REPS),
    ('deepseek-flash-policy',      N_REPS),
    ('qwen2.5:14b',                N_REPS),
    ('qwen2.5:14b-policy',         N_REPS),
]


def load_state():
    if os.path.exists(STATE_JSON):
        return json.load(open(STATE_JSON))
    return {'valid': {}}


def save_state(st):
    json.dump(st, open(STATE_JSON, 'w'), indent=2)


def main():
    st = load_state()
    rc.log('=== run_rssi_batch iniciado ===')
    for name, n_target in PLAN:
        st['valid'].setdefault(name, [])
        rc.log(f'  {name}: alvo {n_target} reps, já tem {len(st["valid"][name])}')

    for name, n_target in PLAN:
        app = APPROACH_BY_NAME[name]
        consecutive_fail = 0
        while len(st['valid'][name]) < n_target:
            rep_num = len(st['valid'][name]) + 1
            got = rc.run_rep(app, rep_num, force_restart=True)
            if got:
                st['valid'][name].append(got)
                save_state(st)
                consecutive_fail = 0
            else:
                consecutive_fail += 1
                if consecutive_fail >= 3:
                    rc.log(f'  !! {name} rep{rep_num}: rejeitada 3x — seguindo pro próximo, tenta depois')
                    break

    # segunda passada pros incompletos
    for name, n_target in PLAN:
        app = APPROACH_BY_NAME[name]
        tries = 0
        while len(st['valid'][name]) < n_target and tries < 3:
            rep_num = len(st['valid'][name]) + 1
            got = rc.run_rep(app, rep_num, force_restart=True)
            tries += 1
            if got:
                st['valid'][name].append(got)
                save_state(st)
                tries = 0
        if len(st['valid'][name]) < n_target:
            rc.log(f'  !! {name}: incompleto ({len(st["valid"][name])}/{n_target}) — desisto')

    rc.log('=== RSSI BATCH CONCLUÍDO ===')
    for name, n_target in PLAN:
        vs = st['valid'].get(name, [])
        if not vs:
            rc.log(f'  {name:<28} 0 reps')
            continue
        losses = [v['loss_pct'] for v in vs]
        lats   = [v['latency_ms'] for v in vs]
        lm = statistics.mean(losses); ls = statistics.pstdev(losses) if len(losses) > 1 else 0
        am = statistics.mean(lats);   as_ = statistics.pstdev(lats) if len(lats) > 1 else 0
        rc.log(f'  {name:<28} n={len(vs)}  loss={lm:.2f}±{ls:.2f}%  lat={am:.1f}±{as_:.1f}ms')
    open('/tmp/exp_logs/RSSI_BATCH_DONE', 'w').close()


if __name__ == '__main__':
    main()
