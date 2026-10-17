#!/usr/bin/env python3


import csv
import json
import os
import re
import socket
import sys
import threading
import time
from math import cos, radians, sqrt
from pathlib import Path

# ── Mininet-WiFi path ────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/wifi/containernet/mininet-wifi/mininet')
sys.path.insert(0, '/home/wifi/containernet/mininet-wifi')

from mn_wifi.net import Mininet_wifi
from mn_wifi.node import OVSKernelAP
from mininet.node import Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

from metrics_collector import MetricsCollector

# ── Config ───────────────────────────────────────────────────────────────────────
ANTENNA_FILE          = Path('data/antenas_regioes_densas.csv')
ANTENNA_SELECTED_FILE = Path('data/antenas_selecionadas_86.json')
DRONE_FILE            = Path('dataset/uas_dataset_final.csv')
LISTEN_PORT           = 5000
MAX_APS               = 86   # igual ao paper (None = todos)
MAX_DRONES            = 50
MONITOR_INTERVAL_S    = 5    # intervalo de ping de monitoramento por drone (s)
HANDOVER_RSSI_MARGIN_DB = 6  # so troca de AP se o ganho de RSSI superar essa margem (evita ping-pong)

# Bottleneck compartilhado — todos os drones disputam esta banda total
SHARED_BW_KBPS        = 30000   # 30 Mbps total — headroom para pings não serem descartados
IPERF_BASE_PORT       = 5201    # porta base; drone slot i usa porta 5201+i

# Coordenada de referência para conversão lat/lon → metros
REF_LAT = 31.22
REF_LON = 121.50

# Potência TX por perfil (dBm)
TX_POWER_DBM = {
    'increase': 20,   # sinal forte — decolagem/voo ativo
    'normal':   15,   # operação padrão
    'reduce':   10,   # sinal reduzido — zona de conflito
}

TX_POWER_QOS = {
    'increase': {'delay_ms': 0.0, 'loss_pct': 0.0},
    'normal':   {'delay_ms': 0.0, 'loss_pct': 0.0},
    'reduce':   {'delay_ms': 0.0, 'loss_pct': 0.0},
}

# Demanda real de tráfego por fase de missão (ground truth, independente do LLM).
# O CBR sender de cada drone é ajustado para este valor quando a intent muda.
# Se alloc (decisão do LLM) < demand → TC descarta excesso → PDR real no sink UDP.
INTENT_DEMAND_KBPS = {
    'I0':    100,   # admin, controle mínimo
    'I1':    500,   # takeoff request: câmera + sensores
    'I2':    300,   # autorização: confirmação
    'I3':   3000,   # em voo: vídeo ao vivo HD + telemetria
    'I4':    800,   # preparação de pouso: câmera de aproximação
    'I5':    300,   # ack de pouso
    'I6':      0,   # pousado (será desativado)
    'I7':    400,   # alerta de conflito
    'I8':    100,   # troca de bateria, solo
    'DEFER': 100,   # aguardando, mobilidade restrita
}

def _channel_params(dist_km: float) -> tuple[float, int]:
    """Modelo log-distance urbano denso (3GPP TR 38.913, UMa, ISD=500-1000m).

    Raio de célula LTE urbano em Shanghai: 500m-1km.
      d < 0.5 km → cobertura excelente (núcleo da célula)
      d 0.5–1.0  → boa cobertura
      d 1.0–1.5  → cobertura aceitável (borda interna)
      d 1.5–2.5  → cobertura degradada (borda da célula)
      d > 2.5    → fora da cobertura nominal
    """
    if dist_km < 0.5:
        return 0.0, 5
    elif dist_km < 1.0:
        return 1.0, 8
    elif dist_km < 1.5:
        return 2.0, 12
    elif dist_km < 2.5:
        return 4.0, 20
    else:
        return 8.0, 35

# ── Coordinate helpers ───────────────────────────────────────────────────────────

def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    """Converte lat/lon para metros a partir da referência REF_LAT/REF_LON."""
    x = (lon - REF_LON) * 111000 * cos(radians(REF_LAT))
    y = (lat - REF_LAT) * 111000
    return round(x, 1), round(y, 1)

def _geo_dist(lat1, lon1, lat2, lon2) -> float:
    """Distância aproximada em km."""
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * cos(radians(lat1))
    return sqrt(dlat**2 + dlon**2)

# ── Data loaders ─────────────────────────────────────────────────────────────────

def load_antennas(antenna_csv, selected_json):
    try:
        selected = set(json.load(open(selected_json)))
        antennas = []
        for r in csv.DictReader(open(antenna_csv, encoding='utf-8')):
            if r['cell'] in selected:
                antennas.append({
                    'lat':      float(r['lat']),
                    'lon':      float(r['lon']),
                    'cell':     r['cell'],
                    'range_m':  int(r.get('range', 1000)),
                })
        if MAX_APS and len(antennas) > MAX_APS:
            # Seleciona MAX_APS APs distribuídos geograficamente (não só os primeiros N)
            antennas.sort(key=lambda a: (a['lat'], a['lon']))
            step = len(antennas) / MAX_APS
            antennas = [antennas[int(i * step)] for i in range(MAX_APS)]
        info(f'[UAS] {len(antennas)} antenas carregadas\n')
        return antennas
    except Exception as e:
        info(f'[warn] {e}\n')
        return [{'lat': 31.22 + i*0.01, 'lon': 121.50 + i*0.01,
                 'cell': str(i), 'range_m': 1000} for i in range(8)]

def load_drones(path, max_drones, antennas):
    try:
        clat = sum(a['lat'] for a in antennas) / len(antennas)
        clon = sum(a['lon'] for a in antennas) / len(antennas)
        all_drones = {}
        for r in csv.DictReader(open(path, encoding='utf-8')):
            s = r.get('sender', '').strip()
            if s and s != 'Control' and s not in all_drones:
                try:
                    all_drones[s] = (float(r['lat']), float(r['lng']))
                except Exception:
                    pass
        nearby = sorted((_geo_dist(clat, clon, lt, ln), s, (lt, ln))
                        for s, (lt, ln) in all_drones.items())
        return {s: pos for _, s, pos in nearby[:max_drones]}
    except Exception as e:
        info(f'[warn] load_drones: {e}\n')
        return {f'DRONE-{i}': (31.22 + i*0.0001, 121.50 + i*0.0001) for i in range(max_drones)}


# ── UAS Network ──────────────────────────────────────────────────────────────────

class UASNetwork:
    """
    Rede Mininet-WiFi com pool dinâmico de estações (drones).
    APs com posição real (Shanghai). TX power em dBm via iw.
    Propagação log-distance (exp=3). Handover por função de custo J.
    """

    DEFAULT_COST_WEIGHTS = (0.2, 0.5, 0.2, 0.1)   # load, dist, interference, handover

    def __init__(self, cost_weights=None):
        self.net        = None
        self.aps        = {}    # ap_name → node
        self.ap_info    = {}    # ap_name → {lat, lon, range, x, y}
        self.stations   = {}    # sta_name → {node, drone_callsign}
        self.active_drones  = {}   # callsign → {sta_name, ap, lat, lon}
        self._drone_stats   = {}   # callsign → metrics dict
        self._ap_handovers  = {}   # ap_name → count
        self._monitor_stop  = {}   # callsign → threading.Event (para parar o monitor)
        self._lock          = threading.Lock()
        self.cost_weights   = cost_weights or self.DEFAULT_COST_WEIGHTS
        self.ground_station = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, antennas, drones):
        info('[UAS] Construindo topologia Mininet-WiFi...\n')
        setLogLevel('info')

        net = Mininet_wifi(controller=Controller, link=TCLink)
        c0  = net.addController('c0')

        # ── Access Points ────────────────────────────────────────────────────
        info('[UAS] Adicionando APs...\n')
        for i, ant in enumerate(antennas, 1):
            ap_name = f'ap{i}'
            x, y    = latlon_to_xy(ant['lat'], ant['lon'])
            ap = net.addAccessPoint(
                ap_name,
                ssid     = 'uas-net',
                mode     = 'g',
                channel  = str((i % 11) + 1),   # canais 1-11 distribuídos
                position = f'{x},{y},10',         # 10m de altura (antena)
                txpower  = 20,                    # 20 dBm = cobertura urbana
                cls      = OVSKernelAP,
                failMode = 'standalone',
            )
            self.aps[ap_name]    = ap
            self.ap_info[ap_name] = {
                'lat': ant['lat'], 'lon': ant['lon'],
                'range': ant['range_m'], 'x': x, 'y': y,
            }

        # ── Ground station (controlador) — addStation para ter wintfs ──────
        gnd_x, gnd_y = latlon_to_xy(REF_LAT, REF_LON)
        self.ground_station = net.addStation(
            'gnd', ip='10.0.0.1/24',
            position=f'{gnd_x},{gnd_y},0',
        )

        # ── Pool de estações (drones inativos) ───────────────────────────────
        info('[UAS] Criando pool de estações...\n')
        ap_list = list(self.aps.keys())
        for idx in range(MAX_DRONES):
            sta_name = f'sta{idx + 1}'
            # Posição inicial fora da área de cobertura (serão movidas na ativação)
            x0 = gnd_x + (idx % 10) * 5
            y0 = gnd_y - 500 - (idx // 10) * 5
            sta = net.addStation(
                sta_name,
                ip       = f'10.0.0.{idx + 20}/24',
                position = f'{x0},{y0},1.5',
                txpower  = 15,
            )
            self.stations[sta_name] = {'node': sta, 'drone_callsign': None}

        # ── Propagação e mobilidade ──────────────────────────────────────────
        net.setPropagationModel(model='logDistance', exp=3)

        info('[UAS] Configurando nós WiFi (hwsim)...\n')
        net.configureNodes()

        info('[UAS] Configurando rede...\n')
        net.build()
        c0.start()
        for ap in self.aps.values():
            ap.start([c0])
        time.sleep(1)

        # ── Switch central para path de dados (wired) ────────────────────────
        # Adicionado APÓS build() para não interferir com configureNodes().
        # Wireless ficam para setTxPower/setPosition. Dados fluem via links TC.
        from mininet.node import OVSSwitch
        self._sw = net.addSwitch('s1', cls=OVSSwitch, failMode='standalone')
        self._sw.start([c0])
        # gnd → switch central (sem bloqueio)
        net.addLink(self.ground_station, self._sw, bw=1000, delay='0ms',
                    cls=TCLink)
        # sta → switch (bloqueado até ativação); porta s1-eth{idx+2}
        for idx in range(MAX_DRONES):
            sta = self.stations[f'sta{idx + 1}']['node']
            net.addLink(sta, self._sw, bw=1, delay='100ms', loss=100,
                        cls=TCLink)
            self.stations[f'sta{idx + 1}']['sw_port'] = f's1-eth{idx + 2}'
        time.sleep(1)

        # Adiciona portas explicitamente ao bridge OVS (addLink não faz isso após build)
        import subprocess
        subprocess.run(['ovs-vsctl', 'add-port', 's1', 's1-eth1'], capture_output=True)
        for idx in range(MAX_DRONES):
            subprocess.run(['ovs-vsctl', 'add-port', 's1', f's1-eth{idx + 2}'],
                           capture_output=True)
        info('[UAS] Portas OVS adicionadas ao bridge s1\n')

        # Configura IPs na subnet 192.168.100.X para path wired de monitoramento
        GND_WIRED_IP = '192.168.100.1'
        self.ground_station.cmd(f'ip addr add {GND_WIRED_IP}/24 dev gnd-eth1')
        for idx in range(MAX_DRONES):
            sta      = self.stations[f'sta{idx + 1}']['node']
            wired_ip = f'192.168.100.{idx + 20}'
            sta.cmd(f'ip addr add {wired_ip}/24 dev {sta.name}-eth1')
            self.stations[f'sta{idx + 1}']['wired_ip'] = wired_ip

        # Receptor UDP — mede latência por timestamp e loss por número de sequência
        udp_sink = '/tmp/udp_sink.py'
        with open(udp_sink, 'w') as _f:
            _f.write(
                'import socket, json, time, threading, struct\n'
                'from collections import defaultdict\n'
                '# Por IP: pkts recebidos, bytes, latências, max_seq visto, gaps\n'
                'stats = defaultdict(lambda: {\n'
                '    "pkts": 0, "bytes": 0,\n'
                '    "lat_sum": 0.0, "lat_count": 0,\n'
                '    "max_seq": -1, "gaps": 0\n'
                '})\n'
                'lock = threading.Lock()\n'
                'def writer():\n'
                '    while True:\n'
                '        time.sleep(2)\n'
                '        with lock:\n'
                '            out = {}\n'
                '            for ip, v in stats.items():\n'
                '                out[ip] = {\n'
                '                    "pkts": v["pkts"],\n'
                '                    "bytes": v["bytes"],\n'
                '                    "avg_latency_ms": round(v["lat_sum"]/v["lat_count"]*1000, 2)\n'
                '                        if v["lat_count"] > 0 else None,\n'
                '                    "loss_gaps": v["gaps"],\n'
                '                    "max_seq": v["max_seq"],\n'
                '                }\n'
                '            with open("/tmp/sink_stats.json","w") as f:\n'
                '                json.dump(out, f)\n'
                'threading.Thread(target=writer, daemon=True).start()\n'
                's = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n'
                "s.bind(('0.0.0.0', 9999))\n"
                'HEADER = struct.Struct("!Id")  # seq(4B unsigned) + timestamp(8B double)\n'
                'HDR_SIZE = HEADER.size\n'
                'while True:\n'
                '    data, addr = s.recvfrom(4096)\n'
                '    t_recv = time.time()\n'
                '    ip = addr[0]\n'
                '    with lock:\n'
                '        st = stats[ip]\n'
                '        st["pkts"] += 1\n'
                '        st["bytes"] += len(data)\n'
                '        if len(data) >= HDR_SIZE:\n'
                '            seq, t_sent = HEADER.unpack_from(data, 0)\n'
                '            latency = t_recv - t_sent\n'
                '            if 0 < latency < 5.0:  # sanity check\n'
                '                st["lat_sum"] += latency\n'
                '                st["lat_count"] += 1\n'
                '            # detecta gaps de sequência (pacotes perdidos)\n'
                '            if st["max_seq"] >= 0 and seq > st["max_seq"] + 1:\n'
                '                st["gaps"] += seq - st["max_seq"] - 1\n'
                '            if seq > st["max_seq"]:\n'
                '                st["max_seq"] = seq\n'
            )
        self.ground_station.cmd(f'nohup python3 {udp_sink} > /dev/null 2>&1 &')

        # Sender CBR: embedar seq_num + send_timestamp para medir latência e loss
        # no próprio fluxo de dados, sem sonda separada.
        udp_sender = '/tmp/udp_sender_cbr.py'
        with open(udp_sender, 'w') as _f:
            _f.write(
                'import socket, sys, time, struct\n'
                'kbps = float(sys.argv[1]) if len(sys.argv) > 1 else 100\n'
                'pkt_size = 1400 if kbps >= 200 else 100\n'
                'HEADER = struct.Struct("!Id")  # seq(4B) + timestamp(8B)\n'
                'pad = b"\\xff" * (pkt_size - HEADER.size)\n'
                'interval = (pkt_size * 8) / (kbps * 1000)\n'
                's = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n'
                "s.connect(('192.168.100.1', 9999))\n"
                'seq = 0\n'
                'while True:\n'
                '    t0 = time.monotonic()\n'
                '    try:\n'
                '        pkt = HEADER.pack(seq, time.time()) + pad\n'
                '        s.send(pkt)\n'
                '        seq += 1\n'
                '    except: pass\n'
                '    rem = interval - (time.monotonic() - t0)\n'
                '    if rem > 0: time.sleep(rem)\n'
            )
        time.sleep(0.5)
        info('[UAS] UDP sink (CBR) ativo em 192.168.100.1:9999\n')

        self.net = net
        info(f'[UAS] ✓ Rede WiFi operacional: {len(self.aps)} APs, '
             f'{len(self.stations)} slots\n')

    # ── Drone lifecycle ───────────────────────────────────────────────────────

    def activate_drone(self, callsign: str, lat: float, lon: float) -> str | None:
        with self._lock:
            if callsign in self.active_drones:
                return self.active_drones[callsign]['sta_name']

            free = next((n for n, d in self.stations.items()
                         if d['drone_callsign'] is None), None)
            if not free:
                info(f'[UAS] POOL CHEIO: sem slot para {callsign}\n')
                return None

            x, y = latlon_to_xy(lat, lon)
            sta  = self.stations[free]['node']
            ap   = self._best_ap_by_rssi(sta, lat, lon)

            # Move estação para posição geográfica real
            sta.setPosition(f'{x},{y},1.5')
            intf_wired   = f'{sta.name}-eth1'
            intf_wlan0   = f'{sta.name}-wlan0'
            sta_n        = int(sta.name.replace('sta', ''))
            sw_port      = f's1-eth{sta_n + 1}'
            default_rate = 1000

            # Calcula loss/delay baseado na distância ao AP mais próximo
            ap_info  = self.ap_info.get(ap, {})
            dist_km  = _geo_dist(lat, lon, ap_info.get('lat', lat),
                                 ap_info.get('lon', lon)) if ap_info else 0.0
            ch_loss, ch_delay = _channel_params(dist_km)
            self._apply_netem(sta, intf_wlan0, intf_wired,
                              default_rate, ch_loss, ch_delay)
            if sw_port:
                self._sw.cmd(f'tc qdisc replace dev {sw_port} root netem')

            self.stations[free]['drone_callsign'] = callsign
            self.active_drones[callsign] = {
                'sta_name': free, 'ap': ap, 'lat': lat, 'lon': lon,
            }

            # Mata senders antigos que possam ter sobrado no slot antes de iniciar novo
            sta.cmd('pkill -f udp_sender_cbr.py 2>/dev/null; sleep 0.1')
            # Inicia sender CBR a 50 kbps (telemetria de localização — demanda base)
            pid_str = sta.cmd('nohup python3 /tmp/udp_sender_cbr.py 50 > /dev/null 2>&1 & echo $!').strip()
            try:
                sender_pid = int(pid_str.split()[-1])
            except Exception:
                sender_pid = None

            _now = time.time()
            self._drone_stats[callsign] = {
                'ap': ap, 'lat': lat, 'lon': lon,
                'wired_ip':             self.stations[free].get('wired_ip', ''),
                'session_start':        _now,
                'session_end':          None,
                'bandwidth_kbps':       1000,  # alocação LLM
                'demand_kbps':          50,    # demanda real (ground truth)
                'demand_history':       [(_now, 50)],  # (timestamp, demand_kbps)
                'latency_ms':           None,
                'packet_loss_pct':      None,
                'session_duration_s':   None,
                'connection_losses':    0,
                'latency_samples':      [],
                'loss_samples':         [],
                'sender_pid':           sender_pid,
                'pkts_expected':        0,
                'pkts_sent_start':      0,
            }
            info(f'[UAS] ↑ {callsign:15} ← {free} @ {ap} ({x:.0f},{y:.0f}) pid={sender_pid}\n')

            # Inicia thread de monitoramento periódico
            stop_evt = threading.Event()
            self._monitor_stop[callsign] = stop_evt
            threading.Thread(
                target=self._monitor_drone,
                args=(callsign, stop_evt),
                daemon=True,
            ).start()

            return free

    def deactivate_drone(self, callsign: str) -> bool:
        # Para o monitor antes de adquirir o lock (monitor também usa lock)
        stop_evt = self._monitor_stop.pop(callsign, None)
        if stop_evt:
            stop_evt.set()

        with self._lock:
            if callsign not in self.active_drones:
                return False

            stats    = self._drone_stats.get(callsign, {})
            duration = time.time() - stats.get('session_start', time.time())
            latency, loss = self._measure_ping(callsign)

            samples   = stats.get('latency_samples', [])
            avg_lat   = round(sum(samples) / len(samples), 3) if samples else round(latency, 3)
            loss_samp = stats.get('loss_samples', [])
            avg_loss  = round(sum(loss_samp) / len(loss_samp), 1) if loss_samp else round(loss, 1)

            self._drone_stats[callsign].update({
                'latency_ms':         avg_lat,
                'packet_loss_pct':    avg_loss,
                'session_duration_s': round(duration, 2),
                'session_end':        time.time(),
                'status':             'landed',
            })

            entry = self.active_drones[callsign]

            # Para o sender UDP do drone (pkill no namespace é mais confiável que kill(pid))
            sta_info_tmp = self.stations[self.active_drones[callsign]['sta_name']]
            sta_info_tmp['node'].cmd('pkill -f udp_sender_cbr.py 2>/dev/null')

            # Bloqueia path de dados wired nos dois lados do veth
            sta_info   = self.stations[entry['sta_name']]
            sta        = sta_info['node']
            intf_wired = f'{sta.name}-eth1'
            sta_n      = int(sta.name.replace('sta', ''))
            sw_port    = f's1-eth{sta_n + 1}'
            sta.cmd(f'tc qdisc replace dev {intf_wired} root netem loss 100%')
            if sw_port:
                self._sw.cmd(f'tc qdisc replace dev {sw_port} root netem loss 100%')
            self.stations[entry['sta_name']]['drone_callsign'] = None
            del self.active_drones[callsign]

            info(f'[UAS] ↓ {callsign:15} lat={latency:.1f}ms '
                 f'loss={loss:.0f}% dur={duration:.0f}s\n')
            return True

    def _monitor_drone(self, callsign: str, stop_evt: threading.Event):
        """Thread periódica: pinga o drone, registra latência/loss, e decide
        handover com base no RSSI real. Roda igual para qualquer drone ativo,
        independente de quem o ativou — mesmo comportamento em todas as
        configurações (baseline e estratégias LLM)."""
        while not stop_evt.wait(timeout=MONITOR_INTERVAL_S):
            with self._lock:
                if callsign not in self.active_drones:
                    break
                latency, loss = self._measure_ping(callsign)
                stats = self._drone_stats.get(callsign)
                if stats is None:
                    break
                stats['latency_samples'].append(latency)
                stats['loss_samples'].append(loss)
                if loss >= 100.0:
                    stats['connection_losses'] += 1
                    info(f'[MON] ✗ {callsign:15} DESCONECTADO '
                         f'(total={stats["connection_losses"]})\n')
                else:
                    info(f'[MON] ✓ {callsign:15} '
                         f'lat={latency:.1f}ms loss={loss:.0f}%\n')

                self._check_rssi_handover(callsign)

    def _check_rssi_handover(self, callsign: str):
        """Compara RSSI real do AP atual com o melhor AP disponível; só troca
        se o ganho superar HANDOVER_RSSI_MARGIN_DB (evita ping-pong por ruído)."""
        entry = self.active_drones.get(callsign)
        if not entry:
            return
        sta = self.stations[entry['sta_name']]['node']
        cur_rssi  = self._rssi(sta, entry['lat'], entry['lon'], entry['ap'])
        best_ap   = self._best_ap_by_rssi(sta, entry['lat'], entry['lon'])
        best_rssi = self._rssi(sta, entry['lat'], entry['lon'], best_ap)
        if best_ap != entry['ap'] and (best_rssi - cur_rssi) > HANDOVER_RSSI_MARGIN_DB:
            info(f'[MON] RSSI {callsign:15} {entry["ap"]}={cur_rssi:.1f}dBm  '
                 f'{best_ap}={best_rssi:.1f}dBm → handover\n')
            self.handover(callsign)

    def handover(self, callsign: str, force_different: bool = False) -> bool:
        if callsign not in self.active_drones:
            return False
        entry  = self.active_drones[callsign]
        old_ap = entry['ap']
        sta    = self.stations[entry['sta_name']]['node']
        exclude = {old_ap} if force_different else None
        new_ap = self._best_ap_by_rssi(sta, entry['lat'], entry['lon'], exclude=exclude)
        if new_ap == old_ap:
            return False

        # Move estação para nova posição e reassocia
        x, y = latlon_to_xy(entry['lat'], entry['lon'])
        sta.setPosition(f'{x},{y},1.5')

        entry['ap'] = new_ap
        if callsign in self._drone_stats:
            self._drone_stats[callsign]['ap'] = new_ap
        self._ap_handovers[new_ap] = self._ap_handovers.get(new_ap, 0) + 1

        # Recalcula channel params para novo AP
        ap_info = self.ap_info.get(new_ap, {})
        dist_km = _geo_dist(entry['lat'], entry['lon'],
                            ap_info.get('lat', entry['lat']),
                            ap_info.get('lon', entry['lon'])) if ap_info else 0.0
        ch_loss, ch_delay = _channel_params(dist_km)
        kbps = self._drone_stats.get(callsign, {}).get('bandwidth_kbps', 1000)
        self._apply_netem(sta,
                          f'{sta.name}-wlan0', f'{sta.name}-eth1',
                          kbps, ch_loss, ch_delay)

        info(f'[UAS] ↔ {callsign:15} handover {old_ap}→{new_ap} '
             f'dist={dist_km:.2f}km loss={ch_loss:.1f}% delay={ch_delay}ms\n')
        return True

    # ── Radio / QoS ───────────────────────────────────────────────────────────

    def set_tx_power(self, callsign: str, level: str) -> bool:
        """Configura potência TX real (dBm via iw)."""
        if callsign not in self.active_drones:
            return False
        dbm = TX_POWER_DBM.get(level, TX_POWER_DBM['normal'])
        sta = self.stations[self.active_drones[callsign]['sta_name']]['node']
        sta.setTxPower(dbm, intf=f"{sta.name}-wlan0")
        return True

    def _apply_netem(self, sta, intf_wlan: str, intf_eth: str,
                     alloc_kbps: int, loss_pct: float, delay_ms: int):
        """Aplica delay + loss + rate limit via netem.
        alloc_kbps = decisão do LLM (limita a banda disponível).
        O sender CBR envia na demanda real (ground truth) — excesso é descartado aqui.
        """
        loss_str  = f'loss {loss_pct:.1f}%' if loss_pct > 0 else ''
        jitter_ms = max(1, int(delay_ms * 0.25)) if delay_ms > 0 else 0
        delay_str = f'delay {delay_ms}ms {jitter_ms}ms distribution normal' if delay_ms > 0 else ''
        rate_str  = f'rate {alloc_kbps}kbit' if alloc_kbps and alloc_kbps > 0 else ''
        parts     = [p for p in [loss_str, delay_str, rate_str] if p]
        params    = ' '.join(parts + ['limit 1000'])
        sta.cmd(f'tc qdisc replace dev {intf_wlan} root netem {params}')
        sta.cmd(f'tc qdisc replace dev {intf_eth}  root netem {params}')

    def _drone_channel(self, callsign: str) -> tuple[float, int]:
        """Calcula loss/delay baseado na distância drone→AP atual."""
        entry   = self.active_drones.get(callsign, {})
        ap_name = entry.get('ap', '')
        ap_info = self.ap_info.get(ap_name, {})
        if not ap_info:
            return 0.0, 5
        dist_km = _geo_dist(entry.get('lat', 0), entry.get('lon', 0),
                            ap_info['lat'], ap_info['lon'])
        return _channel_params(dist_km)

    def set_link_params(self, callsign: str, alloc_kbps: int = None,
                        demand_kbps: int = None,
                        delay_ms: float = None, loss_pct: float = None) -> bool:
        """Configura rede do drone.

        alloc_kbps  — decisão do LLM: limite de banda via tc netem rate.
        demand_kbps — demanda real (ground truth): taxa do CBR sender.
        Se demand > alloc, pacotes excedentes são descartados pelo netem → PDR real.
        """
        if callsign not in self.active_drones:
            return False

        stats     = self._drone_stats.get(callsign, {})
        alloc_kbps = alloc_kbps if alloc_kbps is not None else stats.get('bandwidth_kbps', 1000)
        ch_loss, ch_delay = self._drone_channel(callsign)
        loss_pct  = loss_pct if loss_pct is not None else ch_loss
        delay_ms  = int(delay_ms if delay_ms is not None else ch_delay)

        sta_name  = self.active_drones[callsign]['sta_name']
        sta       = self.stations[sta_name]['node']
        intf_wlan = f'{sta.name}-wlan0'
        intf_eth  = f'{sta.name}-eth1'

        try:
            # Aplica limite de banda (alocação LLM) + canal (loss/delay) — só refaz o
            # tc qdisc replace se a alocação pedida for realmente diferente da atual;
            # evita resetar a fila da interface (e descartar pacotes em trânsito) por
            # reconfigurações redundantes, mantendo a demanda real sempre atualizada
            # (abaixo) independente disso.
            if alloc_kbps != stats.get('bandwidth_kbps'):
                self._apply_netem(sta, intf_wlan, intf_eth, alloc_kbps, loss_pct, delay_ms)
                self._drone_stats[callsign]['bandwidth_kbps'] = alloc_kbps

            # Reinicia sender só se demanda real mudou
            if demand_kbps is not None and demand_kbps != stats.get('demand_kbps'):
                import subprocess as _sp
                old_pid = self._drone_stats[callsign].get('sender_pid')
                if old_pid:
                    _sp.run(['kill', str(old_pid)], capture_output=True)
                pid_str = sta.cmd(
                    f'nohup python3 /tmp/udp_sender_cbr.py {demand_kbps} > /dev/null 2>&1 & echo $!'
                ).strip()
                try:
                    self._drone_stats[callsign]['sender_pid'] = int(pid_str.split()[-1])
                except Exception:
                    pass
                self._drone_stats[callsign]['demand_kbps'] = demand_kbps
                self._drone_stats[callsign].setdefault('demand_history', []).append(
                    (time.time(), demand_kbps)
                )

            info(f'[UAS] ⊙ {callsign:15} alloc={alloc_kbps}kbps demand={self._drone_stats[callsign].get("demand_kbps","?")}kbps loss={loss_pct:.1f}% delay={delay_ms}ms\n')
            return True
        except Exception as e:
            info(f'[UAS] set_link_params error: {e}\n')
            return False

    def set_bandwidth(self, callsign: str, alloc_kbps: int) -> bool:
        return self.set_link_params(callsign, alloc_kbps=alloc_kbps)

    def set_demand(self, callsign: str, demand_kbps: int) -> bool:
        """Atualiza a taxa do CBR sender (demanda real do drone, ground truth)."""
        return self.set_link_params(callsign, demand_kbps=demand_kbps)

    # ── Measurement ───────────────────────────────────────────────────────────

    def _measure_ping(self, callsign: str) -> tuple[float, float]:
        """Mede RTT e perda via path wired: ground station → drone (5 pings)."""
        entry = self.active_drones.get(callsign)
        if not entry or not self.ground_station:
            return 2.0, 0.0
        sta_info = self.stations[entry['sta_name']]
        drone_ip = sta_info.get('wired_ip') or sta_info['node'].IP()
        result   = self.ground_station.cmd(
            f'ping -c 3 -i 0.2 -W 2 {drone_ip} 2>&1')
        info(f'[DBG] ping {drone_ip}: {result[:120]}\n')
        loss_m  = re.search(r'(\d+(?:\.\d+)?)% packet loss', result)
        rtt_m   = re.search(r'rtt .+= [\d.]+/([\d.]+)/', result)
        latency = float(rtt_m.group(1))  if rtt_m  else 2.0
        loss    = float(loss_m.group(1)) if loss_m else 100.0
        return latency, loss

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            'aps':           len(self.aps),
            'stations_total': len(self.stations),
            'drones_active': len(self.active_drones),
            'active_drones': {
                cs: {'sta': d['sta_name'], 'ap': d['ap'],
                     'lat': d['lat'], 'lon': d['lon']}
                for cs, d in self.active_drones.items()
            },
        }

    def reset_all(self):
        """Desativa todos os drones ativos — limpa estado entre runs de modelos."""
        for callsign in list(self.active_drones.keys()):
            stop_evt = self._monitor_stop.pop(callsign, None)
            if stop_evt:
                stop_evt.set()
        with self._lock:
            for callsign, entry in list(self.active_drones.items()):
                sta_info   = self.stations[entry['sta_name']]
                sta        = sta_info['node']
                intf_wired = f'{sta.name}-eth1'
                sta_n      = int(sta.name.replace('sta', ''))
                sw_port    = f's1-eth{sta_n + 1}'
                sta.cmd('pkill -f udp_sender_cbr.py 2>/dev/null')
                sta.cmd(f'tc qdisc replace dev {intf_wired} root netem loss 100%')
                self._sw.cmd(f'tc qdisc replace dev {sw_port} root netem loss 100%')
                self.stations[entry['sta_name']]['drone_callsign'] = None
            self.active_drones.clear()
            self._drone_stats.clear()
            self._ap_handovers.clear()
        # Garante que todos os senders zumbis sejam eliminados (pkill host-level como root)
        import subprocess as _sp
        _sp.run(['pkill', '-f', 'udp_sender_cbr.py'], capture_output=True)
        info('[UAS] reset_all: pool limpo\n')

    def _get_sink_stats(self) -> dict:
        """Lê métricas UDP do sink: pkts, latência média, gaps de sequência."""
        try:
            with open('/tmp/sink_stats.json') as f:
                return json.load(f)
        except Exception:
            return {}

    def _reset_sink_stats(self):
        """Reinicia o processo sink para zerar contadores em memória entre runs."""
        import subprocess as _sp
        _sp.run(['pkill', '-f', 'udp_sink.py'], capture_output=True)
        time.sleep(0.5)
        self.ground_station.cmd('nohup python3 /tmp/udp_sink.py > /dev/null 2>&1 &')
        time.sleep(0.5)

    def save_metrics(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = f'results/network_metrics_{int(time.time())}.json'

        for callsign in list(self.active_drones.keys()):
            stats = self._drone_stats.get(callsign, {})
            dur   = time.time() - stats.get('session_start', time.time())
            samp  = stats.get('latency_samples', [])
            lsamp = stats.get('loss_samples', [])
            self._drone_stats[callsign].update({
                'latency_ms':         round(sum(samp)/len(samp), 3) if samp else None,
                'packet_loss_pct':    round(sum(lsamp)/len(lsamp), 1) if lsamp else None,
                'session_duration_s': round(dur, 2),
                'status':             'active',
            })

        ap_load = {}
        for s in self._drone_stats.values():
            ap_load[s['ap']] = ap_load.get(s['ap'], 0) + 1

        samples_all  = [v for s in self._drone_stats.values()
                         for v in s.get('latency_samples', [])]
        loss_all     = [v for s in self._drone_stats.values()
                         for v in s.get('loss_samples', [])]
        total_losses = sum(s.get('connection_losses', 0)
                           for s in self._drone_stats.values())

        # PDR e latência UDP: derivados do próprio fluxo CBR (seq_num + timestamp)
        sink = self._get_sink_stats()
        udp_loss_all = []
        udp_lat_all  = []
        for cs, s in self._drone_stats.items():
            drone_ip  = s.get('wired_ip', '')
            entry     = sink.get(drone_ip, {})
            received  = entry.get('pkts', 0)
            gaps      = entry.get('loss_gaps', 0)       # pkts perdidos detectados
            sink_lat  = entry.get('avg_latency_ms')     # latência do fluxo CBR

            history   = s.get('demand_history', [])
            t_end     = s.get('session_end') or time.time()

            # Pacotes esperados via demand_history (ground truth)
            expected = 0
            for i, (t_start, demand) in enumerate(history):
                t_phase_end = history[i + 1][0] if i + 1 < len(history) else t_end
                phase_dur   = max(0.0, t_phase_end - t_start)
                pkt_size    = 1400 if demand >= 200 else 100
                expected   += (demand * 1000 * phase_dur) / (8 * pkt_size)
            expected = max(1, int(expected))

            # Loss: usa gaps de sequência se disponível, senão recebidos vs esperados
            if gaps > 0 or received > 0:
                total_sent = received + gaps
                udp_loss = max(0.0, round(gaps / max(1, total_sent) * 100, 2))
            else:
                udp_loss = max(0.0, round((1 - min(1.0, received / expected)) * 100, 2))

            s['udp_pkts_received'] = received
            s['udp_pkts_expected'] = expected
            s['udp_loss_gaps']     = gaps
            s['udp_loss_pct']      = udp_loss
            s['udp_latency_ms']    = sink_lat
            udp_loss_all.append(udp_loss)
            if sink_lat is not None:
                udp_lat_all.append(sink_lat)

        output = {
            'summary': {
                'total_drones_activated':  len(self._drone_stats),
                'drones_active_at_end':    len(self.active_drones),
                'total_handovers':         sum(self._ap_handovers.values()),
                'aps_used':                len(ap_load),
                'ap_load_distribution':    ap_load,
                'total_connection_losses': total_losses,
                # Primary metrics: UDP CBR flow (OWD from embedded timestamps, loss from seq gaps)
                'avg_latency_ms':          round(sum(udp_lat_all) / len(udp_lat_all), 3)
                                           if udp_lat_all else None,
                'avg_packet_loss_pct':     round(sum(udp_loss_all) / len(udp_loss_all), 2)
                                           if udp_loss_all else None,
                # Secondary: ping RTT (used only for connection loss detection)
                'avg_ping_rtt_ms':         round(sum(samples_all) / len(samples_all), 3)
                                           if samples_all else None,
                'avg_ping_loss_pct':       round(sum(loss_all) / len(loss_all), 2)
                                           if loss_all else None,
            },
            'drones': {
                cs: {
                    'ap':                 s['ap'],
                    'lat':                s['lat'],
                    'lon':                s['lon'],
                    'alloc_kbps':         s['bandwidth_kbps'],
                    'demand_kbps':        s.get('demand_kbps', 50),
                    'session_duration_s': s.get('session_duration_s'),
                    'udp_latency_ms':     s.get('udp_latency_ms'),   # OWD from CBR flow
                    'udp_loss_pct':       s.get('udp_loss_pct'),     # seq-gap based
                    'udp_pkts_received':  s.get('udp_pkts_received', 0),
                    'udp_pkts_expected':  s.get('udp_pkts_expected', 0),
                    'udp_loss_gaps':      s.get('udp_loss_gaps', 0),
                    'ping_latency_ms':    s.get('latency_ms'),        # ping RTT (reference)
                    'ping_loss_pct':      s.get('packet_loss_pct'),
                    'connection_losses':  s.get('connection_losses', 0),
                    'status':             s.get('status', 'landed'),
                }
                for cs, s in self._drone_stats.items()
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        info(f'[UAS] Métricas → {filepath}\n')
        return filepath

    # ── AP selection ──────────────────────────────────────────────────────────

    def _rssi(self, sta, lat: float, lon: float, ap_name: str) -> float:
        """RSSI real (dBm) via modelo de propagação do Mininet-WiFi (logDistance,
        já configurado em build()). Depende só da potência do AP e da distância —
        a potência do próprio drone não entra na conta (downlink)."""
        ap_info  = self.ap_info[ap_name]
        dist_m   = _geo_dist(lat, lon, ap_info['lat'], ap_info['lon']) * 1000
        sta_intf = sta.wintfs[0]
        ap_intf  = self.aps[ap_name].wintfs[0]
        return sta_intf.get_rssi(ap_intf, dist_m)

    def _best_ap_by_rssi(self, sta, lat: float, lon: float,
                          exclude: set | None = None) -> str:
        if not self.ap_info:
            return list(self.aps.keys())[0]
        candidates = [n for n in self.ap_info if not exclude or n not in exclude]
        if not candidates:
            candidates = list(self.ap_info.keys())
        rssi   = {n: self._rssi(sta, lat, lon, n) for n in candidates}
        worst, best = min(rssi.values()), max(rssi.values())
        spread = (best - worst) or 1.0
        # norm_badness: 0 = melhor sinal do conjunto, 1 = pior
        norm_badness = {n: (best - rssi[n]) / spread for n in candidates}
        return min(candidates, key=lambda n: self._ap_cost(n, norm_badness[n]))

    def _ap_cost(self, ap_name: str, norm_dist: float) -> float:
        """J = α·load + β·dist + γ·interference + δ·handover"""
        alpha, beta, gamma, delta = self.cost_weights
        AP_CAPACITY         = 10
        INTERFERENCE_RANGE  = 0.5   # km

        load_count = sum(1 for d in self.active_drones.values()
                         if d['ap'] == ap_name)
        load = max(0.0, (load_count - AP_CAPACITY // 2) / (AP_CAPACITY // 2))

        ap = self.ap_info[ap_name]
        n_interfering = sum(
            1 for n, o in self.ap_info.items()
            if n != ap_name
            and _geo_dist(ap['lat'], ap['lon'], o['lat'], o['lon']) < INTERFERENCE_RANGE
        )
        interference  = n_interfering / max(len(self.ap_info) - 1, 1)
        max_h         = max(self._ap_handovers.values(), default=1)
        handover_cost = self._ap_handovers.get(ap_name, 0) / max_h if max_h else 0.0

        return (alpha * load + beta * norm_dist
                + gamma * interference + delta * handover_cost)


# ── TCP Server ────────────────────────────────────────────────────────────────────

def serve(uas: UASNetwork, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(20)
    info(f'[Server] Ouvindo :{port}\n')
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_handle, args=(conn, addr, uas),
                         daemon=True).start()


def _handle(conn, addr, uas):
    try:
        conn.settimeout(1)
        data = conn.recv(4096)
        if not data:
            return
        cmd      = json.loads(data.decode().strip())
        action   = cmd.get('action', '')
        drone    = cmd.get('drone', '')
        radio    = cmd.get('radio_access', {})
        is_bcast = cmd.get('broadcast', False)

        info(f'[Server] ← {cmd.get("intent",""):10} {drone:15} {action}\n')

        if action == 'activate':
            lat  = float(cmd.get('lat', REF_LAT))
            lon  = float(cmd.get('lon', REF_LON))
            uas.activate_drone(drone, lat, lon)
            kbps = int(cmd.get('bandwidth_kbps', 0))
            tx   = radio.get('tx_power', 'increase')
            if kbps > 0:
                uas.set_bandwidth(drone, kbps)
            uas.set_tx_power(drone, tx)
            demand = cmd.get('demand_kbps')
            if demand is not None and int(demand) > 0:
                uas.set_demand(drone, int(demand))

        elif action == 'deactivate':
            uas.deactivate_drone(drone)

        elif action == 'set_demand':
            uas.set_demand(drone, int(cmd.get('demand_kbps', 50)))

        elif action == 'bandwidth':
            delay_ms   = cmd.get('delay_ms')
            demand_kbps = cmd.get('demand_kbps')
            uas.set_link_params(drone,
                                alloc_kbps=int(cmd.get('bandwidth_kbps', 0)),
                                demand_kbps=int(demand_kbps) if demand_kbps is not None else None,
                                delay_ms=float(delay_ms) if delay_ms is not None else None)

        elif action == 'reset_sink':
            uas._reset_sink_stats()

        elif action == 'handover':
            lat = cmd.get('lat')
            lon = cmd.get('lon')
            if lat and lon and drone in uas.active_drones:
                uas.active_drones[drone].update(
                    {'lat': float(lat), 'lon': float(lon)})
                if drone in uas._drone_stats:
                    uas._drone_stats[drone].update(
                        {'lat': float(lat), 'lon': float(lon)})
            demand = cmd.get('demand_kbps')
            if demand is not None and int(demand) > 0:
                uas.set_demand(drone, int(demand))
            uas.handover(drone, force_different=bool(cmd.get('force')))

        elif action == 'status':
            conn.send(json.dumps(uas.get_status()).encode())

        elif action == 'save_metrics':
            path = uas.save_metrics()
            conn.send(json.dumps({'saved': path}).encode())

        elif action == 'reset_all':
            uas.reset_all()
            conn.send(json.dumps({'reset': True}).encode())

        elif action == 'kill_senders':
            import subprocess as _sp
            r = _sp.run(['pkill', '-f', 'udp_sender_cbr.py'], capture_output=True)
            killed = r.returncode == 0
            # também mata dentro de cada namespace via sta.cmd
            for sta_info in uas.stations.values():
                sta_info['node'].cmd('pkill -f udp_sender_cbr.py 2>/dev/null')
            conn.send(json.dumps({'killed': killed}).encode())

        if radio.get('tx_power'):
            targets = (list(uas.active_drones.keys()) if is_bcast
                       else ([drone] if drone else []))
            for cs in targets:
                uas.set_tx_power(cs, radio['tx_power'])

    except Exception as e:
        info(f'[Server] erro: {e}\n')
    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    setLogLevel('info')
    antennas = load_antennas(ANTENNA_FILE, ANTENNA_SELECTED_FILE)
    drones   = load_drones(DRONE_FILE, MAX_DRONES, antennas)

    info(f'[UAS] Iniciando: {len(antennas)} APs, {len(drones)} drones\n')

    uas = UASNetwork()
    uas.build(antennas, drones)

    threading.Thread(target=serve, args=(uas, LISTEN_PORT),
                     daemon=True).start()

    info(f'[UAS] Pronto! TCP :{LISTEN_PORT}. Ctrl+C para encerrar.\n\n')

    try:
        if sys.stdin.isatty():
            CLI(uas.net)
        else:
            import signal
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        uas.net.stop()


if __name__ == '__main__':
    main()
