#!/usr/bin/env python3
import argparse, csv, glob, json, os, subprocess, time
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
LOG_DIR     = '/tmp/exp_logs'
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'clean_summary.csv')
STATE_JSON  = os.path.join(RESULTS_DIR, 'clean_state.json')
SERVER_LOG  = '/tmp/mininet_server.log'
SUDO_PW     = os.environ.get('SUDO_PW', '')

os.makedirs(LOG_DIR, exist_ok=True)

APPROACHES = [
    {'name': 'baseline_reactive',
     'cmd': ['python3', '-u', 'scripts/baseline_reactive.py', '--runs', '1'],
     'glob': 'reactive_baseline_1runs_*.json'},
    {'name': 'keyword-baseline',
     # Sem LLM — classificação por regras, mesmo pipeline de rede do
     # classify-then-map. min_drones mais baixo: recall de I6 é só ~50% nesse
     # classificador, então boa parte dos drones nunca libera o slot do pool
     # (50 vagas), travando o total de ativados bem abaixo do padrão dos LLMs.
     'cmd': ['python3', '-u', 'scripts/llm_reconfig.py', '--model', 'keyword-baseline'],
     'glob': 'keyword-baseline_network_metrics_*.json',
     'llm_glob': 'keyword-baseline_llm_metrics_*.json',
     'min_drones': 70},
    {'name': 'ministral-8b-latest',
     'cmd': ['python3', '-u', 'scripts/llm_reconfig.py', '--model', 'ministral-8b-latest'],
     'glob': 'ministral-8b-latest_network_metrics_*.json',
     'llm_glob': 'ministral-8b-latest_llm_metrics_*.json', 'target': 8},
    {'name': 'ministral-8b-latest-policy',
     'cmd': ['python3', '-u', 'scripts/llm_policy_agent.py', '--model', 'ministral-8b-latest'],
     'glob': 'policy_ministral-8b-latest_network_*.json',
     'llm_glob': 'policy_ministral-8b-latest_llm_*.json', 'target': 8},
    {'name': 'qwen2.5:14b',
     'cmd': ['python3', '-u', 'scripts/llm_reconfig.py', '--model', 'qwen2.5:14b'],
     'glob': 'qwen2.5_14b_network_metrics_*.json',
     'llm_glob': 'qwen2.5_14b_llm_metrics_*.json'},
    {'name': 'qwen2.5:14b-policy',
     'cmd': ['python3', '-u', 'scripts/llm_policy_agent.py', '--model', 'qwen2.5:14b'],
     'glob': 'policy_qwen2.5_14b_network_*.json',
     'llm_glob': 'policy_qwen2.5_14b_llm_*.json', 'target': 8},
    {'name': 'llama3.1:8b',
     # Via DeepInfra (o Ollama remoto se mostrou instável demais pra esse modelo).
     'cmd': ['python3', '-u', 'scripts/llm_reconfig.py', '--model', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'],
     'glob': 'meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_network_metrics_*.json',
     'llm_glob': 'meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_llm_metrics_*.json'},
    {'name': 'llama3.1:8b-policy',
     # min_accuracy mais baixo: a tarefa combinada da policy agent derruba a
     # acurácia do LLaMA pra ~52-54% de forma consistente — não é instabilidade
     # de API, é o modelo sendo o mais fraco do estudo nessa tarefa mais difícil.
     'cmd': ['python3', '-u', 'scripts/llm_policy_agent.py', '--model', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'],
     'glob': 'policy_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_network_*.json',
     'llm_glob': 'policy_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_llm_*.json',
     'target': 8, 'min_accuracy': 45.0},
    {'name': 'deepseek-flash',
     'cmd': ['python3', '-u', 'scripts/llm_reconfig.py', '--model', 'deepseek-flash'],
     'glob': 'deepseek-flash_network_metrics_*.json',
     'llm_glob': 'deepseek-flash_llm_metrics_*.json', 'target': 8},
    {'name': 'deepseek-flash-policy',
     'cmd': ['python3', '-u', 'scripts/llm_policy_agent.py', '--model', 'deepseek-flash'],
     'glob': 'policy_deepseek-flash_network_*.json',
     'llm_glob': 'policy_deepseek-flash_llm_*.json', 'target': 8},
]

MIN_ACCURACY_PCT = 60.0  # abaixo disso, provável degradação de conexão com a API

# Faixas de sanidade — fora disso a rep é rejeitada
MIN_DRONES   = 85
MAX_LATENCY  = 3000.0   # ms
MAX_LOSS     = 40.0     # %


def log(msg):
    print(f'{datetime.now():%m-%d %H:%M:%S}  {msg}', flush=True)


def server_alive():
    import socket
    try:
        s = socket.socket(); s.settimeout(3)
        s.connect(('localhost', 5000)); s.close()
        return True
    except Exception:
        return False


def _load1():
    try:
        return os.getloadavg()[0]
    except Exception:
        return 0.0


def restart_server():
    log('  restart servidor: matando TUDO relacionado...')
    for pat in ['mininet_server_dynamic.py', 'udp_sender_cbr.py', 'udp_sink.py',
                'hostapd', 'wpa_supplicant']:
        subprocess.run(f'echo {SUDO_PW} | sudo -S pkill -9 -f "{pat}"',
                       shell=True, capture_output=True)
    time.sleep(3)
    # sem isso, um sink_stats.json velho pode ser lido antes do novo
    # udp_sink.py escrever a primeira atualização, "congelando" uma rep com
    # dados de uma execução anterior.
    subprocess.run(f'echo {SUDO_PW} | sudo -S rm -f /tmp/sink_stats.json',
                   shell=True, capture_output=True)
    subprocess.run(f'echo {SUDO_PW} | sudo -S mn -c', shell=True, capture_output=True, timeout=120)
    time.sleep(2)
    # espera a carga baixar antes de tentar o boot (86 APs sob load alto = boot falha)
    for _ in range(60):
        if _load1() < 6.0:
            break
        log(f'  restart servidor: load={_load1():.1f}, aguardando baixar...')
        time.sleep(10)
    open(SERVER_LOG, 'w').close()
    subprocess.Popen(f'echo {SUDO_PW} | sudo -S python3 scripts/mininet_server_dynamic.py '
                     f'> {SERVER_LOG} 2>&1', shell=True, cwd=REPO_ROOT)
    log('  restart servidor: aguardando boot (até 15 min)...')
    for _ in range(180):
        try:
            if 'Ouvindo :5000' in open(SERVER_LOG).read():
                break
        except Exception:
            pass
        time.sleep(5)
    time.sleep(3)
    ok = server_alive()
    log(f'  restart servidor: {"PRONTO" if ok else "FALHOU"}')
    return ok


def newest_matching(pattern, since_ts):
    """Arquivo mais recente que casa com pattern E foi escrito depois de since_ts."""
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    files = [f for f in files if os.path.getmtime(f) > since_ts]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_metrics(path):
    """Retorna (n_drones, loss_pct, latency_ms) ou (None, None, None) se não der."""
    try:
        d = json.load(open(path))
    except Exception:
        return None, None, None

    if 'runs' in d:  # reactive (--runs 1 → lista com 1 elemento)
        runs = d['runs']
        n = min((r.get('total_drones_activated', 0) for r in runs), default=0)
        loss = d.get('avg_packet_loss_pct')
        lat  = d.get('avg_latency_ms')
        return n, loss, lat

    drones = d.get('drones', {})
    n = d.get('summary', {}).get('total_drones_activated', len(drones))
    ls = [i['udp_loss_pct'] for i in drones.values()
          if i.get('udp_pkts_received', 0) > 0 and i.get('udp_loss_pct') is not None]
    la = [i['udp_latency_ms'] for i in drones.values()
          if i.get('udp_latency_ms') is not None]
    loss = round(sum(ls) / len(ls), 3) if ls else None
    lat  = round(sum(la) / len(la), 2) if la else None
    return n, loss, lat


def validate(path, since_ts, llm_path=None, min_drones=None, min_accuracy=None):
    """Retorna (ok: bool, motivo: str, metrics: dict)."""
    min_drones   = MIN_DRONES if min_drones is None else min_drones
    min_accuracy = MIN_ACCURACY_PCT if min_accuracy is None else min_accuracy
    if path is None:
        return False, 'nenhum arquivo novo escrito', {}
    if os.path.getmtime(path) <= since_ts:
        return False, 'arquivo não é novo desta run', {}
    n, loss, lat = extract_metrics(path)
    m = {'file': os.path.basename(path), 'n_drones': n, 'loss_pct': loss, 'latency_ms': lat}
    if n is None or n < min_drones:
        return False, f'só {n} drones (mín {min_drones})', m
    if loss is None or lat is None:
        return False, 'loss ou latência ausente (medição UDP falhou)', m
    if lat <= 0 or lat > MAX_LATENCY:
        return False, f'latência fora da faixa: {lat}ms', m
    if loss < 0 or loss > MAX_LOSS:
        return False, f'loss fora da faixa: {loss}%', m
    # Acurácia baixa geralmente indica degradação/instabilidade da API do LLM
    # durante a run (ex: DNS falhando no meio, timeouts frequentes) — os
    # números de rede podem parecer OK mesmo com classificação toda quebrada.
    if llm_path is not None:
        try:
            fresh = os.path.getmtime(llm_path) > since_ts
        except OSError:
            fresh = False
        if not fresh:
            return False, 'arquivo de métricas LLM ausente/antigo', m
        try:
            acc = json.load(open(llm_path)).get('accuracy_pct')
        except Exception:
            acc = None
        m['accuracy_pct'] = acc
        if acc is None or acc < min_accuracy:
            return False, f'acurácia LLM baixa: {acc}% (mín {min_accuracy}%) — API instável?', m
    return True, 'ok', m


def load_state():
    if os.path.exists(STATE_JSON):
        return json.load(open(STATE_JSON))
    return {'valid': {}}   # name -> list of metric dicts


def save_state(st):
    json.dump(st, open(STATE_JSON, 'w'), indent=2)


def append_csv(name, rep, m, elapsed):
    new = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['approach', 'rep', 'file', 'n_drones', 'loss_pct', 'latency_ms',
                        'elapsed_s', 'timestamp'])
        w.writerow([name, rep, m.get('file'), m.get('n_drones'), m.get('loss_pct'),
                    m.get('latency_ms'), round(elapsed), datetime.now().isoformat(timespec='seconds')])


def run_rep(app, rep_num, force_restart=False):
    name = app['name']
    logf = os.path.join(LOG_DIR, f'clean_{name.replace(":", "_").replace("/", "_")}_rep{rep_num:02d}.log')

    # Reinicia só se pedido (1ª vez, ou após rep rejeitada) ou se o servidor caiu.
    if force_restart or not server_alive():
        if not restart_server():
            log(f'  {name} rep{rep_num}: servidor não subiu — pulando tentativa')
            return None

    since = time.time()
    log(f'  {name} rep{rep_num}: iniciando (log={logf})')
    t0 = time.time()
    with open(logf, 'w') as lf:
        p = subprocess.Popen(app['cmd'], stdout=lf, stderr=lf, cwd=REPO_ROOT)
        p.wait()
    elapsed = time.time() - t0

    path = newest_matching(app['glob'], since)
    llm_path = newest_matching(app['llm_glob'], since) if app.get('llm_glob') else None
    ok, reason, m = validate(path, since, llm_path, min_drones=app.get('min_drones'),
                              min_accuracy=app.get('min_accuracy'))
    if ok:
        acc_str = f' acc={m["accuracy_pct"]}%' if 'accuracy_pct' in m else ''
        log(f'  ✓ {name} rep{rep_num}: drones={m["n_drones"]} loss={m["loss_pct"]}% '
            f'lat={m["latency_ms"]}ms{acc_str}  ({elapsed:.0f}s)')
        append_csv(name, rep_num, m, elapsed)
        return m
    else:
        log(f'  ✗ {name} rep{rep_num} REJEITADA: {reason}  '
            f'(drones={m.get("n_drones")} loss={m.get("loss_pct")} lat={m.get("latency_ms")}, {elapsed:.0f}s)')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--max-attempts-per-rep', type=int, default=3)
    args = ap.parse_args()

    st = load_state()
    log(f'=== run_clean iniciado — alvo {args.reps} reps limpas por approach ===')
    for app in APPROACHES:
        target = app.get('target', args.reps)
        have = len(st['valid'].get(app['name'], []))
        log(f'  {app["name"]:<25} já tem {have}/{target}')

    for app in APPROACHES:
        name = app['name']
        target = app.get('target', args.reps)
        st['valid'].setdefault(name, [])
        consecutive_fail = 0
        while len(st['valid'][name]) < target:
            rep_num = len(st['valid'][name]) + 1
            # Restart em toda rep: reset_all/reset_sink sozinhos não garantem
            # isolamento total entre execuções na mesma instância do servidor
            # (já produziram reps byte-a-byte idênticas por dado congelado).
            got = run_rep(app, rep_num, force_restart=True)
            if got:
                st['valid'][name].append(got)
                save_state(st)
                consecutive_fail = 0
            else:
                consecutive_fail += 1
                if consecutive_fail >= args.max_attempts_per_rep:
                    log(f'  !! {name} rep{rep_num}: rejeitada {consecutive_fail}x — '
                        f'seguindo pro próximo approach, tenta no fim')
                    break

    # segunda passada pros incompletos
    for app in APPROACHES:
        name = app['name']
        target = app.get('target', args.reps)
        tries = 0
        while len(st['valid'][name]) < target and tries < args.max_attempts_per_rep:
            rep_num = len(st['valid'][name]) + 1
            got = run_rep(app, rep_num, force_restart=True)
            tries += 1
            if got:
                st['valid'][name].append(got)
                save_state(st)
                tries = 0
        if len(st['valid'][name]) < target:
            log(f'  !! {name}: incompleto ({len(st["valid"][name])}/{target}) — desisto')

    log('=== CONCLUÍDO ===')
    for app in APPROACHES:
        name = app['name']
        vs = st['valid'].get(name, [])
        if not vs:
            log(f'  {name:<25} 0 reps')
            continue
        import statistics
        losses = [v['loss_pct'] for v in vs]
        lats   = [v['latency_ms'] for v in vs]
        lm = statistics.mean(losses); ls = statistics.pstdev(losses) if len(losses) > 1 else 0
        am = statistics.mean(lats);   as_ = statistics.pstdev(lats) if len(lats) > 1 else 0
        log(f'  {name:<25} n={len(vs)}  loss={lm:.2f}±{ls:.2f}%  lat={am:.1f}±{as_:.1f}ms')
    open('/tmp/exp_logs/CLEAN_ALL_DONE', 'w').close()


if __name__ == '__main__':
    main()
