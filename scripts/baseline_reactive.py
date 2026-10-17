#!/usr/bin/env python3

import csv, json, socket, time, threading
from datetime import datetime
from pathlib import Path
from trajectory_interp import interpolate_trajectories, trajectory_stats

# ── Config ────────────────────────────────────────────────────────────────────
FIXED_BW          = 1000    # kbps inicial
RIC_HOST          = 'localhost'
RIC_PORT          = 5000
CSV_FILE          = Path('dataset/uas_dataset_final.csv')
RESULTS_DIR       = Path('results')
MSG_DELAY         = 0.5     # segundos entre mensagens

MONITOR_INTERVAL_S = 30     # igual ao servidor
LATENCY_THRESHOLD  = 18.0   # ms — acima disso, reduz banda
LOSS_THRESHOLD     = 2.0    # % — acima disso, reduz banda
BW_STEP_DOWN       = 0.80   # reduz para 80% da banda atual
BW_STEP_UP         = 1.20   # sobe para 120% (recuperação)
BW_MIN             = 500    # kbps mínimo
BW_MAX             = 2000   # kbps máximo

# Demanda real por intent (ground truth) — CBR sender ajusta taxa na mudança de fase.
INTENT_DEMAND_KBPS = {
    'I0': 100, 'I1': 500, 'I2': 300, 'I3': 3000,
    'I4': 800, 'I5': 300, 'I6': 0,   'I7': 400, 'I8': 100,
}

# ── Estado local ──────────────────────────────────────────────────────────────
activated    = {}   # callsign → {'bw': int, 'lat': float, 'lon': float}
bw_changes   = []   # log de ajustes de banda reativos
stats_lock   = threading.Lock()

# ── TCP helper ────────────────────────────────────────────────────────────────
def send_cmd(cmd: dict) -> dict | None:
    try:
        s = socket.socket()
        s.settimeout(8)
        s.connect((RIC_HOST, RIC_PORT))
        s.sendall(json.dumps(cmd).encode() + b'\n')
        try:
            resp = s.recv(65536)
            return json.loads(resp) if resp else None
        except Exception:
            return None
        finally:
            s.close()
    except Exception as e:
        print(f'  [TCP error] {e}')
        return None

def send_only(cmd: dict) -> bool:
    try:
        s = socket.socket()
        s.settimeout(5)
        s.connect((RIC_HOST, RIC_PORT))
        s.sendall(json.dumps(cmd).encode() + b'\n')
        s.close()
        return True
    except Exception as e:
        print(f'  [TCP error] {e}')
        return False

# ── Monitor reativo ───────────────────────────────────────────────────────────
def reactive_monitor(stop_evt: threading.Event, network_changes: list):
    """Thread que consulta métricas e ajusta banda reativamente."""
    while not stop_evt.wait(timeout=MONITOR_INTERVAL_S):
        status = send_cmd({'command': 'network_config', 'action': 'status',
                           'drone': '', 'intent': ''})
        if not status:
            continue

        with stats_lock:
            active_now = dict(activated)

        for callsign, local in active_now.items():
            drone_status = status.get('active_drones', {}).get(callsign, {})
            lat_ms = drone_status.get('latency_ms')
            loss   = drone_status.get('packet_loss_pct')
            if lat_ms is None:
                continue

            current_bw = local['bw']
            new_bw     = current_bw

            if lat_ms > LATENCY_THRESHOLD or (loss and loss > LOSS_THRESHOLD):
                # Degradação detectada → reduz banda reativamente
                new_bw = max(BW_MIN, int(current_bw * BW_STEP_DOWN))
                reason = f'lat={lat_ms:.1f}ms loss={loss:.1f}%'
            elif lat_ms < LATENCY_THRESHOLD * 0.6 and (not loss or loss < LOSS_THRESHOLD * 0.5):
                # Link bom → tenta recuperar banda
                new_bw = min(BW_MAX, int(current_bw * BW_STEP_UP))
                reason = f'recovery lat={lat_ms:.1f}ms'
            else:
                continue

            if new_bw != current_bw:
                send_only({'command': 'network_config', 'action': 'bandwidth',
                           'drone': callsign, 'intent': 'reactive_bw',
                           'bandwidth_kbps': new_bw})
                with stats_lock:
                    if callsign in activated:
                        activated[callsign]['bw'] = new_bw
                bw_changes.append({'callsign': callsign, 'old_bw': current_bw,
                                   'new_bw': new_bw, 'reason': reason,
                                   'ts': time.time()})
                print(f'  [REACTIVE] {callsign}: {current_bw}→{new_bw}kbps ({reason})')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ts_run = int(time.time())

    # Limpa estado do servidor entre runs
    send_cmd({'command': 'network_config', 'action': 'reset_all', 'drone': '', 'intent': ''})
    time.sleep(1)
    send_cmd({'command': 'network_config', 'action': 'reset_sink', 'drone': '', 'intent': ''})
    time.sleep(1)

    messages = []
    with open(CSV_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                messages.append({
                    'timestamp': datetime.strptime(row['timestamp'][:19], '%Y-%m-%d %H:%M:%S'),
                    'sender':    row['sender'].strip(),
                    'intent':    row['intent'].strip(),
                    'lat':       float(row.get('lat') or 0),
                    'lon':       float(row.get('lng') or 0),
                })
            except Exception:
                pass

    messages = interpolate_trajectories(messages)
    tstats = trajectory_stats(messages)
    print(f'Mensagens carregadas: {len(messages)} | '
          f'trajetórias: {tstats["drones_with_trajectory"]} drones, '
          f'dist. média={tstats["avg_trajectory_length_m"]}m')
    print(f'Baseline REATIVA (SSF): bw_inicial={FIXED_BW}kbps, '
          f'latency_thr={LATENCY_THRESHOLD}ms, loss_thr={LOSS_THRESHOLD}%\n')

    network_changes = []
    stop_monitor    = threading.Event()
    monitor_thread  = threading.Thread(
        target=reactive_monitor, args=(stop_monitor, network_changes), daemon=True)
    monitor_thread.start()

    for i, msg in enumerate(messages, 1):
        intent = msg['intent']
        sender = msg['sender']
        lat, lon = msg['lat'], msg['lon']

        print(f'  [{i:4}/{len(messages)}] {sender:15} [{intent}]', end='\r')

        if intent == 'I1' and sender not in activated:
            # Ativa drone — banda fixa inicial (NÃO pré-configura por I2)
            ok = send_only({
                'command': 'network_config', 'action': 'activate',
                'intent': 'I1_reactive', 'drone': sender,
                'lat': lat, 'lon': lon,
                'bandwidth_kbps': FIXED_BW,
                'demand_kbps': INTENT_DEMAND_KBPS.get(intent, 100),
                'network_control': {'bandwidth_kbps': FIXED_BW, 'qos_class': 'BE'},
            })
            if ok:
                with stats_lock:
                    activated[sender] = {'bw': FIXED_BW, 'lat': lat, 'lon': lon}
                network_changes.append({'msg_idx': i, 'sender': sender,
                                        'intent': intent, 'action': 'activate',
                                        'bandwidth_kbps': FIXED_BW})

        elif sender in activated:
            if intent == 'I6':
                # Pouso — desativa
                send_only({'command': 'network_config', 'action': 'deactivate',
                           'intent': 'I6_reactive', 'drone': sender})
                with stats_lock:
                    activated.pop(sender, None)
                network_changes.append({'msg_idx': i, 'sender': sender,
                                        'intent': intent, 'action': 'deactivate'})
            else:
                # Qualquer mensagem do drone ativo: atualiza posição + demanda real
                with stats_lock:
                    if sender in activated:
                        activated[sender].update({'lat': lat, 'lon': lon})
                send_only({'command': 'network_config', 'action': 'handover',
                           'drone': sender, 'intent': intent,
                           'lat': lat, 'lon': lon,
                           'demand_kbps': INTENT_DEMAND_KBPS.get(intent, 100)})

        time.sleep(MSG_DELAY)

    print(f'\nProcessamento concluído. Parando monitor e salvando métricas...')
    stop_monitor.set()
    monitor_thread.join(timeout=5)

    send_cmd({'command': 'network_config', 'action': 'save_metrics',
              'drone': '', 'intent': ''})
    time.sleep(3)

    # Coleta resultado do servidor
    net_files = sorted(RESULTS_DIR.glob('network_metrics_*.json'))
    if net_files:
        net_data = json.load(open(net_files[-1]))
        net_data['summary']['mode']                = 'reactive_ssf'
        net_data['summary']['initial_bw_kbps']     = FIXED_BW
        net_data['summary']['latency_threshold_ms'] = LATENCY_THRESHOLD
        net_data['summary']['loss_threshold_pct']   = LOSS_THRESHOLD
        net_data['summary']['reactive_bw_changes']  = len(bw_changes)
        out = RESULTS_DIR / f'reactive_network_metrics_{ts_run}.json'
        json.dump(net_data, open(out, 'w'), indent=2)
        print(f'Métricas de rede → {out}')
        s = net_data['summary']
        print(f"  drones: {s['total_drones_activated']}  "
              f"handovers: {s['total_handovers']}  "
              f"bw_changes: {len(bw_changes)}  "
              f"latency: {s['avg_latency_ms']}ms  "
              f"loss: {s['avg_packet_loss_pct']}%")

    llm_path = RESULTS_DIR / f'reactive_llm_metrics_{ts_run}.json'
    json.dump({'mode': 'reactive_ssf', 'total_messages': len(messages),
               'initial_bw_kbps': FIXED_BW, 'reactive_bw_changes': bw_changes,
               'network_changes': network_changes},
              open(llm_path, 'w'), indent=2)
    print(f'Log reativo → {llm_path}')

def run_once(messages):
    """Executa um replay completo do baseline reativo e retorna métricas."""
    global activated, bw_changes
    activated  = {}
    bw_changes = []

    send_cmd({'command': 'network_config', 'action': 'reset_all', 'drone': '', 'intent': ''})
    time.sleep(1)
    send_cmd({'command': 'network_config', 'action': 'reset_sink', 'drone': '', 'intent': ''})
    time.sleep(1)

    network_changes = []
    stop_monitor    = threading.Event()
    monitor_thread  = threading.Thread(
        target=reactive_monitor, args=(stop_monitor, network_changes), daemon=True)
    monitor_thread.start()

    for i, msg in enumerate(messages, 1):
        intent = msg['intent']
        sender = msg['sender']
        lat, lon = msg['lat'], msg['lon']

        print(f'  [{i:4}/{len(messages)}] {sender:15} [{intent}]', end='\r')

        if intent == 'I1' and sender not in activated:
            ok = send_only({
                'command': 'network_config', 'action': 'activate',
                'intent': 'I1_reactive', 'drone': sender,
                'lat': lat, 'lon': lon,
                'bandwidth_kbps': FIXED_BW,
                'demand_kbps': INTENT_DEMAND_KBPS.get(intent, 100),
                'network_control': {'bandwidth_kbps': FIXED_BW, 'qos_class': 'BE'},
            })
            if ok:
                with stats_lock:
                    activated[sender] = {'bw': FIXED_BW, 'lat': lat, 'lon': lon}
        elif sender in activated:
            if intent == 'I6':
                send_only({'command': 'network_config', 'action': 'deactivate',
                           'intent': 'I6_reactive', 'drone': sender})
                with stats_lock:
                    activated.pop(sender, None)
            else:
                with stats_lock:
                    if sender in activated:
                        activated[sender].update({'lat': lat, 'lon': lon})
                send_only({'command': 'network_config', 'action': 'handover',
                           'drone': sender, 'intent': intent,
                           'lat': lat, 'lon': lon,
                           'demand_kbps': INTENT_DEMAND_KBPS.get(intent, 100)})
        time.sleep(MSG_DELAY)

    print()
    stop_monitor.set()
    monitor_thread.join(timeout=5)
    time.sleep(30)

    send_cmd({'command': 'network_config', 'action': 'save_metrics', 'drone': '', 'intent': ''})
    time.sleep(3)

    net_files = sorted(RESULTS_DIR.glob('network_metrics_*.json'))
    if not net_files:
        return None
    s = json.load(open(net_files[-1])).get('summary', {})
    print(f"  loss={s.get('avg_packet_loss_pct','?')}%  lat={s.get('avg_latency_ms','?')}ms  "
          f"handovers={s.get('total_handovers','?')}")
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    messages = []
    with open(CSV_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                messages.append({
                    'sender':  row['sender'].strip(),
                    'intent':  row['intent'].strip(),
                    'lat':     float(row.get('lat') or 0),
                    'lon':     float(row.get('lng') or 0),
                })
            except Exception:
                pass
    print(f'Mensagens: {len(messages)} | Runs: {args.runs}')

    ckpt_path = RESULTS_DIR / f'reactive_baseline_checkpoint.json'
    results   = []
    start_run = 1
    if args.resume and ckpt_path.exists():
        ckpt      = json.load(open(ckpt_path))
        results   = ckpt.get('runs', [])
        start_run = len(results) + 1
        print(f'[RESUME] {len(results)} runs já concluídas, continuando da {start_run}')

    from statistics import mean, stdev
    for r in range(start_run, args.runs + 1):
        print(f'\n--- Repetição {r}/{args.runs} ---')
        s = run_once(messages)
        if s:
            results.append(s)
            ckpt_data = {
                'mode': 'reactive_ssf', 'n_runs': len(results),
                'avg_packet_loss_pct': round(mean(x['avg_packet_loss_pct'] for x in results if x.get('avg_packet_loss_pct') is not None), 3),
                'std_packet_loss_pct': round(stdev([x['avg_packet_loss_pct'] for x in results if x.get('avg_packet_loss_pct') is not None]), 3) if len(results) > 1 else 0.0,
                'avg_latency_ms': round(mean(x['avg_latency_ms'] for x in results if x.get('avg_latency_ms') is not None), 3),
                'std_latency_ms': round(stdev([x['avg_latency_ms'] for x in results if x.get('avg_latency_ms') is not None]), 3) if len(results) > 1 else 0.0,
                'avg_handovers': round(mean(x['total_handovers'] for x in results if x.get('total_handovers') is not None), 3),
                'std_handovers': round(stdev([x['total_handovers'] for x in results if x.get('total_handovers') is not None]), 3) if len(results) > 1 else 0.0,
                'runs': results,
            }
            json.dump(ckpt_data, open(ckpt_path, 'w'), indent=2)
            print(f'  [checkpoint] {len(results)}/{args.runs} runs salvos')

    ts  = int(time.time())
    out = RESULTS_DIR / f'reactive_baseline_{args.runs}runs_{ts}.json'
    json.dump(ckpt_data, open(out, 'w'), indent=2)
    ckpt_path.unlink(missing_ok=True)
    print(f'\n=== RESULTADO FINAL ===')
    print(f'  Loss:      {ckpt_data["avg_packet_loss_pct"]} ± {ckpt_data["std_packet_loss_pct"]}%')
    print(f'  Latência:  {ckpt_data["avg_latency_ms"]} ± {ckpt_data["std_latency_ms"]}ms')
    print(f'  Handovers: {ckpt_data["avg_handovers"]} ± {ckpt_data["std_handovers"]}')
    print(f'  → {out}')


if __name__ == '__main__':
    main()
