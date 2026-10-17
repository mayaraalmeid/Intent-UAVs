"""
Trajectory interpolation for sparse drone GPS datasets.

Problem: the synthetic dataset has GPS coordinates only at takeoff (I1) and
landing (I4/I6). Messages in between repeat the takeoff position, causing
zero-distance handover evaluations.

Fix: for each drone, linearly interpolate lat/lon between consecutive GPS
anchor points so every message carries a unique, gradually-changing position.
This makes handover evaluation meaningful without altering GT intents or
adding synthetic messages.
"""
from math import sqrt, cos, radians
from collections import defaultdict


def interpolate_trajectories(messages: list[dict]) -> list[dict]:
    """
    Returns a copy of `messages` with lat/lon interpolated for each drone.

    Algorithm:
      1. Group message indices by sender (pilots only, not Control).
      2. Find GPS "anchor" points — messages where lat/lon is non-zero AND
         differs from the previous anchor by >10 m (avoids treating duplicate
         GPS as separate anchors).
      3. Between any two consecutive anchors, distribute intermediate positions
         linearly along the great-circle chord.
      4. Messages before the first anchor or after the last keep their original
         GPS (or 0,0 if they had none).
    """
    result = [m.copy() for m in messages]

    # Group by sender
    by_sender: dict[str, list[int]] = defaultdict(list)
    for i, msg in enumerate(messages):
        if msg.get('sender', '').strip() not in ('Control', ''):
            by_sender[msg['sender'].strip()].append(i)

    MIN_ANCHOR_DIST_M = 10.0  # ignore duplicate GPS jitter

    for sender, idxs in by_sender.items():
        # Collect anchor points: (message_index, lat, lon)
        anchors: list[tuple[int, float, float]] = []
        for i in idxs:
            la, lo = messages[i].get('lat', 0.0), messages[i].get('lon', 0.0)
            if la == 0.0 and lo == 0.0:
                continue
            if anchors:
                prev_la, prev_lo = anchors[-1][1], anchors[-1][2]
                dist = _dist_m(prev_la, prev_lo, la, lo)
                if dist < MIN_ANCHOR_DIST_M:
                    continue  # same location — skip duplicate
            anchors.append((i, la, lo))

        if len(anchors) < 2:
            continue  # nothing to interpolate

        # Interpolate between each pair of consecutive anchors
        for seg in range(len(anchors) - 1):
            a_idx, a_lat, a_lon = anchors[seg]
            b_idx, b_lat, b_lon = anchors[seg + 1]

            # All sender message indices in [a_idx, b_idx] inclusive
            seg_idxs = [i for i in idxs if a_idx <= i <= b_idx]
            n = len(seg_idxs)
            if n < 2:
                continue

            for k, msg_i in enumerate(seg_idxs):
                t = k / (n - 1)
                result[msg_i]['lat'] = a_lat + t * (b_lat - a_lat)
                result[msg_i]['lon'] = a_lon + t * (b_lon - a_lon)

    return result


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * 111_000 * cos(radians((lat1 + lat2) / 2))
    return sqrt(dlat ** 2 + dlon ** 2)


def trajectory_stats(messages: list[dict]) -> dict:
    """Return per-drone movement stats for logging."""
    by_sender: dict[str, list] = defaultdict(list)
    for msg in messages:
        s = msg.get('sender', '').strip()
        la, lo = msg.get('lat', 0.0), msg.get('lon', 0.0)
        if s not in ('Control', '') and (la or lo):
            by_sender[s].append((la, lo))

    total_dist, n_drones = 0.0, 0
    for pts in by_sender.values():
        if len(pts) < 2:
            continue
        n_drones += 1
        dist = sum(_dist_m(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                   for i in range(len(pts)-1))
        total_dist += dist

    return {
        'drones_with_trajectory': n_drones,
        'avg_trajectory_length_m': round(total_dist / n_drones, 1) if n_drones else 0,
        'total_distance_km': round(total_dist / 1000, 2),
    }
