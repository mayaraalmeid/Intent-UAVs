
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from metrics_framework import ExperimentMetrics, NetworkMetrics, LLMMetrics


class MetricsCollector:
    """Simplified interface for collecting experiment metrics."""
    
    def __init__(self, experiment_id: str, description: str = "", output_dir: str = 'results'):
        self.experiment_id = experiment_id
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metric containers
        self.network_metrics = NetworkMetrics()
        self.llm_metrics = LLMMetrics()
        
        print(f"[COLLECTOR] Initialized: {experiment_id}")
    
    # ── Network Metrics ─────────────────────────────────────────────────────────
    
    def record_drone_latency(self, drone: str, latency_ms: float, intent: Optional[str] = None, ap_id: Optional[str] = None):
        """Record latency measurement for a drone."""
        self.network_metrics.record_latency(drone, latency_ms, intent, ap_id)
    
    def record_drone_packet_loss(self, drone: str, loss_pct: float, packets_sent: int = 0, packets_lost: int = 0, intent: Optional[str] = None):
        """Record packet loss for a drone."""
        self.network_metrics.record_packet_loss(drone, loss_pct, packets_sent, packets_lost, intent)
    
    def record_drone_session(self, drone: str, duration_sec: float, intent_sequence: list = None, handovers: int = 0):
        """Record a connection session for a drone."""
        self.network_metrics.record_connection_time(drone, duration_sec, intent_sequence, handovers)
    
    # ── LLM Metrics ────────────────────────────────────────────────────────────
    
    def record_llm_prediction(self, model: str, message: str, predicted: str, actual: str, confidence: float, latency_ms: float):
        """Record a single LLM prediction."""
        self.llm_metrics.record_prediction(model, message, predicted, actual, confidence, latency_ms)
    
    # ── Batch Recording ────────────────────────────────────────────────────────
    
    def record_batch_llm_predictions(self, model: str, predictions: list):
        """Record multiple predictions at once.
        
        predictions format:
        [
            {'message': 'text', 'predicted': 'I1', 'actual': 'I1', 'confidence': 0.98, 'latency_ms': 234},
            ...
        ]
        """
        for pred in predictions:
            self.llm_metrics.record_prediction(
                model=model,
                message=pred['message'],
                predicted_intent=pred['predicted'],
                ground_truth=pred['actual'],
                confidence=pred['confidence'],
                latency_ms=pred['latency_ms']
            )
    
    def record_batch_drone_latencies(self, drone: str, latency_samples: list):
        
        for sample in latency_samples:
            self.network_metrics.record_latency(
                drone,
                sample['latency_ms'],
                sample.get('intent'),
                sample.get('ap_id')
            )
    
    # ── Export & Save ───────────────────────────────────────────────────────────
    
    def save(self, filename: Optional[str] = None, show_summary: bool = True) -> str:
        """Save all metrics to JSON file."""
        if filename is None:
            filename = f'metrics_{self.experiment_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        filepath = self.output_dir / filename
        
        # Create experiment container
        exp = ExperimentMetrics(self.experiment_id, self.description)
        exp.add_network_metrics(self.network_metrics)
        exp.add_llm_metrics(self.llm_metrics)
        
        # Save to JSON
        saved_path = exp.save(str(filepath), pretty=True)
        
        if show_summary:
            exp.display_summary()
        
        return saved_path
    
    def save_csv(self, prefix: str = ""):
        """Export metrics to CSV files (optional)."""
        csv_dir = self.output_dir / f"{prefix}{self.experiment_id}_csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Network metrics CSV
        net_rows = []
        for drone_id, metrics in self.network_metrics.drones.items():
            for lat_sample in metrics.latencies:
                net_rows.append({
                    'drone': drone_id,
                    'type': 'latency',
                    'value': lat_sample.latency_ms,
                    'unit': 'ms',
                    'intent': lat_sample.intent,
                    'timestamp': lat_sample.timestamp,
                })
            for loss_sample in metrics.packet_losses:
                net_rows.append({
                    'drone': drone_id,
                    'type': 'packet_loss',
                    'value': loss_sample.loss_pct,
                    'unit': '%',
                    'intent': loss_sample.intent,
                    'timestamp': loss_sample.timestamp,
                })
        
        if net_rows:
            import csv
            net_csv = csv_dir / 'network_metrics.csv'
            with open(net_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['drone', 'type', 'value', 'unit', 'intent', 'timestamp'])
                writer.writeheader()
                writer.writerows(net_rows)
            print(f"[CSV] Saved {net_csv}")
        
        # LLM metrics CSV
        llm_rows = []
        for model_name, model_metrics in self.llm_metrics.models.items():
            for pred in model_metrics.predictions:
                llm_rows.append({
                    'model': model_name,
                    'message': pred.message[:100],  # Truncate for CSV
                    'predicted': pred.predicted_intent,
                    'actual': pred.ground_truth,
                    'correct': 'yes' if pred.correct else 'no',
                    'confidence': pred.confidence,
                    'latency_ms': pred.latency_ms,
                    'timestamp': pred.timestamp,
                })
        
        if llm_rows:
            import csv
            llm_csv = csv_dir / 'llm_metrics.csv'
            with open(llm_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['model', 'message', 'predicted', 'actual', 'correct', 'confidence', 'latency_ms', 'timestamp'])
                writer.writeheader()
                writer.writerows(llm_rows)
            print(f"[CSV] Saved {llm_csv}")
        
        return csv_dir
    
    def summary(self) -> dict:
        """Get summary without saving."""
        exp = ExperimentMetrics(self.experiment_id, self.description)
        exp.add_network_metrics(self.network_metrics)
        exp.add_llm_metrics(self.llm_metrics)
        return exp.to_dict()
    
    def print_summary(self):
        """Print summary to console."""
        exp = ExperimentMetrics(self.experiment_id, self.description)
        exp.add_network_metrics(self.network_metrics)
        exp.add_llm_metrics(self.llm_metrics)
        exp.display_summary()


# ── Convenience Functions ──────────────────────────────────────────────────────

def create_collector(experiment_id: str, description: str = "") -> MetricsCollector:
    """Factory function to create a metrics collector."""
    return MetricsCollector(experiment_id, description)


def compare_experiments(exp1_path: str, exp2_path: str) -> dict:
    """Compare two saved experiment results."""
    with open(exp1_path) as f:
        exp1 = json.load(f)
    with open(exp2_path) as f:
        exp2 = json.load(f)
    
    comparison = {
        'experiment_1': exp1['experiment']['id'],
        'experiment_2': exp2['experiment']['id'],
    }
    
    # LLM comparison
    if 'llm' in exp1 and 'llm' in exp2:
        models1 = {m: s['accuracy'] for m, s in exp1['llm']['models'].items()}
        models2 = {m: s['accuracy'] for m, s in exp2['llm']['models'].items()}
        
        comparison['llm_accuracy'] = {
            'exp1': models1,
            'exp2': models2,
        }
    
    return comparison


# ── Example Usage ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("[EXAMPLE] MetricsCollector Demo\n")
    
    # Create collector
    collector = MetricsCollector(
        experiment_id='demo_001',
        description='Testing Claude vs GPT on UAS intents'
    )
    
    # 1. Record network metrics
    print("Recording network metrics...")
    collector.record_drone_latency('DRONE-22', 45.3, intent='I2', ap_id='ap1')
    collector.record_drone_latency('DRONE-22', 42.1, intent='I2', ap_id='ap1')
    collector.record_drone_latency('DRONE-22', 51.2, intent='I2', ap_id='ap1')
    collector.record_drone_packet_loss('DRONE-22', 0.5, 1000, 5, intent='I2')
    collector.record_drone_session('DRONE-22', 120.5, ['I1', 'I2', 'I6'], handovers=0)
    
    collector.record_drone_latency('DRONE-27', 38.2, intent='I2', ap_id='ap2')
    collector.record_drone_latency('DRONE-27', 39.5, intent='I2', ap_id='ap2')
    collector.record_drone_session('DRONE-27', 95.2, ['I1', 'I2', 'I3', 'I6'], handovers=1)
    
    # 2. Record LLM metrics
    print("Recording LLM predictions...")
    
    # Batch: Claude predictions
    claude_preds = [
        {'message': 'DRONE-22 requesting takeoff', 'predicted': 'I1', 'actual': 'I1', 'confidence': 0.98, 'latency_ms': 234},
        {'message': 'DRONE-22 now in zone Y', 'predicted': 'I3', 'actual': 'I3', 'confidence': 0.95, 'latency_ms': 198},
        {'message': 'cleared for takeoff', 'predicted': 'I2', 'actual': 'I2', 'confidence': 0.99, 'latency_ms': 156},
        {'message': 'requesting landing', 'predicted': 'I4', 'actual': 'I4', 'confidence': 0.97, 'latency_ms': 212},
        {'message': 'landed safely', 'predicted': 'I6', 'actual': 'I6', 'confidence': 0.99, 'latency_ms': 189},
    ]
    collector.record_batch_llm_predictions('claude', claude_preds)
    
    # Batch: GPT predictions
    gpt_preds = [
        {'message': 'DRONE-22 requesting takeoff', 'predicted': 'I1', 'actual': 'I1', 'confidence': 0.94, 'latency_ms': 189},
        {'message': 'DRONE-22 now in zone Y', 'predicted': 'I3', 'actual': 'I3', 'confidence': 0.89, 'latency_ms': 167},
        {'message': 'cleared for takeoff', 'predicted': 'I2', 'actual': 'I2', 'confidence': 0.96, 'latency_ms': 145},
        {'message': 'requesting landing', 'predicted': 'I4', 'actual': 'I4', 'confidence': 0.91, 'latency_ms': 201},
        {'message': 'landed safely', 'predicted': 'I6', 'actual': 'I6', 'confidence': 0.95, 'latency_ms': 178},
    ]
    collector.record_batch_llm_predictions('gpt', gpt_preds)
    
    # 3. Save and display
    print("\nSaving metrics...")
    collector.save()
    
    # 4. Also save as CSV
    collector.save_csv()
    
    print("\n✅ Demo complete!")
