# Brion Quantum - Google TPU Supercomputer AI Framework v2.0
# TPU-optimized orchestration for quantum-classical hybrid workloads
# Supports Google TRC Program TPU allocations (v4, v5e, v6e)

import time
import random
import logging
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('TPU_SC_AI')


# ============================================================================
# TPU Resource Definitions (Google TRC Program Allocations)
# ============================================================================

@dataclass
class TPUAllocation:
    """Represents a Google TRC TPU allocation."""
    zone: str
    chip_type: str  # v4, v5e, v6e
    chip_count: int
    mode: str  # 'spot' or 'on-demand'
    flops_per_chip: float = 0.0  # PFLOPS

    def __post_init__(self):
        flops_map = {'v4': 275.0, 'v5e': 393.0, 'v6e': 918.0}
        self.flops_per_chip = flops_map.get(self.chip_type, 100.0)

    @property
    def total_tflops(self):
        return self.chip_count * self.flops_per_chip

    def to_dict(self):
        return {
            'zone': self.zone,
            'chip_type': self.chip_type,
            'chips': self.chip_count,
            'mode': self.mode,
            'total_tflops': self.total_tflops,
        }


# Brion Quantum's TRC allocations
TRC_ALLOCATIONS = [
    TPUAllocation('europe-west4-a', 'v6e', 64, 'spot'),
    TPUAllocation('us-central2-b', 'v4', 32, 'on-demand'),
    TPUAllocation('europe-west4-b', 'v5e', 64, 'spot'),
    TPUAllocation('us-east1-d', 'v6e', 64, 'spot'),
    TPUAllocation('us-central1-a', 'v5e', 64, 'spot'),
    TPUAllocation('us-central2-b', 'v4', 32, 'spot'),
]


@dataclass
class ComputeNode:
    """Represents a TPU or ASIC compute node."""
    id: int
    node_type: str = 'tpu'
    chip_type: str = 'v6e'
    hash_rate: float = 0.0
    temp: float = 65.0
    power: float = 300.0
    utilization: float = 0.0
    status: str = 'idle'
    tflops: float = 918.0
    error_count: int = 0
    uptime: float = 0.0

    def __post_init__(self):
        if self.node_type == 'miner':
            self.hash_rate = random.uniform(90, 110)
            self.temp = random.uniform(60, 90)
            self.power = random.uniform(2500, 3500)
        else:
            flops_map = {'v4': 275.0, 'v5e': 393.0, 'v6e': 918.0}
            self.tflops = flops_map.get(self.chip_type, 100.0)

    def get_status(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.node_type,
            'chip': self.chip_type,
            'hash_rate': self.hash_rate,
            'temp': self.temp,
            'power': self.power,
            'utilization': self.utilization,
            'tflops': self.tflops,
            'status': self.status,
            'errors': self.error_count,
            'uptime': self.uptime,
        }

    def apply_optimization(self, new_freq, new_voltage):
        self.hash_rate *= new_freq / 100
        self.power *= new_voltage / 100
        self.temp += random.uniform(-2, 2)
        self.temp = max(30, min(95, self.temp))
        return True


class TPUWorkloadScheduler:
    """
    Intelligent workload scheduler for TPU clusters.
    Routes jobs to optimal TPU zones based on latency, chip type, and availability.
    """

    def __init__(self, allocations: List[TPUAllocation] = None):
        self.allocations = allocations or TRC_ALLOCATIONS
        self.job_queue: deque = deque()
        self.running_jobs: Dict[int, Dict] = {}
        self.completed_jobs: List[Dict] = []
        self.job_counter = 0

        # Chips currently committed per allocation, keyed by allocation identity.
        # Scheduling previously compared a request against an allocation's total
        # chip count and never decremented anything, so the same 64 chips could
        # be handed to an unlimited number of jobs at once.
        self._in_use: Dict[int, int] = {id(a): 0 for a in self.allocations}

    # -- Capacity -----------------------------------------------------------

    def available_chips(self, allocation: TPUAllocation) -> int:
        """Chips in this allocation not currently committed to a job."""
        return allocation.chip_count - self._in_use.get(id(allocation), 0)

    def capacity_report(self) -> List[Dict[str, Any]]:
        """Per-allocation capacity and utilization."""
        report = []
        for a in self.allocations:
            used = self._in_use.get(id(a), 0)
            report.append({
                'zone': a.zone,
                'chip_type': a.chip_type,
                'total_chips': a.chip_count,
                'chips_in_use': used,
                'chips_available': a.chip_count - used,
                'utilization': used / a.chip_count if a.chip_count else 0.0,
            })
        return report

    # -- Job lifecycle ------------------------------------------------------

    def submit_job(self, job_type: str, required_chips: int, preferred_chip: str = 'v6e') -> Dict:
        """
        Submit a compute job.

        A job that cannot be placed right now stays queued rather than being
        reported as scheduled, and is retried whenever capacity is released.
        """
        if required_chips <= 0:
            raise ValueError("required_chips must be positive")

        self.job_counter += 1
        job = {
            'id': self.job_counter,
            'type': job_type,
            'required_chips': required_chips,
            'preferred_chip': preferred_chip,
            'status': 'queued',
            'submitted_at': time.time(),
            'assigned_zone': None,
            'assigned_chip': None,
        }

        if not self._place(job):
            self.job_queue.append(job)
            logger.warning(
                f"Job {job['id']} queued - no allocation with "
                f"{required_chips} free chips"
            )
        return job

    def complete_job(self, job_id: int) -> Optional[Dict]:
        """
        Mark a running job finished, releasing its chips.

        Freed capacity is immediately offered to queued jobs, so a queue that
        was blocked on capacity drains without another submit.
        """
        job = self.running_jobs.pop(job_id, None)
        if job is None:
            return None

        allocation = job.pop('_allocation', None)
        if allocation is not None:
            key = id(allocation)
            self._in_use[key] = max(0, self._in_use.get(key, 0) - job['required_chips'])

        job['status'] = 'completed'
        job['completed_at'] = time.time()
        job['duration'] = job['completed_at'] - job['submitted_at']
        self.completed_jobs.append(job)

        self._drain_queue()
        return job

    def _drain_queue(self) -> List[Dict]:
        """Place as many queued jobs as freed capacity now allows, in order."""
        placed, deferred = [], []
        while self.job_queue:
            job = self.job_queue.popleft()
            if self._place(job):
                placed.append(job)
            else:
                deferred.append(job)
        # Preserve submission order for everything still unplaceable.
        self.job_queue.extend(deferred)
        return placed

    def _place(self, job: Dict) -> bool:
        """Commit a job to the best allocation with room, if any."""
        allocation = self._find_best_allocation(
            job['required_chips'], job['preferred_chip']
        )
        if allocation is None:
            return False

        self._in_use[id(allocation)] = (
            self._in_use.get(id(allocation), 0) + job['required_chips']
        )
        job['assigned_zone'] = allocation.zone
        job['assigned_chip'] = allocation.chip_type
        job['status'] = 'scheduled'
        job['scheduled_at'] = time.time()
        job['_allocation'] = allocation
        self.running_jobs[job['id']] = job
        logger.info(
            f"Job {job['id']} scheduled on {allocation.chip_type} in {allocation.zone}"
        )
        return True

    def _find_best_allocation(self, required_chips: int, preferred_chip: str) -> Optional[TPUAllocation]:
        """
        Best allocation with enough *free* chips.

        Ties break toward the allocation with the least spare capacity, so a
        small job does not consume the one zone large enough for a big one.
        """
        candidates = [a for a in self.allocations
                      if self.available_chips(a) >= required_chips]
        if not candidates:
            return None

        preferred = [a for a in candidates if a.chip_type == preferred_chip]
        pool = preferred or candidates
        return min(pool, key=lambda a: (self.available_chips(a), -a.flops_per_chip))

    def get_total_compute(self) -> Dict[str, Any]:
        """Get total compute capacity across all allocations."""
        total_chips = sum(a.chip_count for a in self.allocations)
        total_tflops = sum(a.total_tflops for a in self.allocations)
        chips_in_use = sum(self._in_use.values())
        return {
            'total_chips': total_chips,
            'chips_in_use': chips_in_use,
            'chips_available': total_chips - chips_in_use,
            'utilization': chips_in_use / total_chips if total_chips else 0.0,
            'total_tflops': total_tflops,
            'total_pflops': total_tflops / 1000,
            'allocations': len(self.allocations),
            'jobs_queued': len(self.job_queue),
            'jobs_running': len(self.running_jobs),
            'jobs_completed': len(self.completed_jobs),
        }


class ASICCommanderAI:
    """
    Brion Quantum AI Commander v2.0

    Autonomous optimization engine for TPU/ASIC compute clusters.
    Features:
    - Adaptive learning with reward-based optimization
    - Thermal management with predictive cooling
    - Efficiency scoring with Pareto-optimal targeting
    - TPU workload scheduling integration
    - Performance history with trend analysis
    """

    VERSION = "2.0.4"

    def __init__(self, nodes: List[ComputeNode] = None, scheduler: TPUWorkloadScheduler = None):
        self.nodes = nodes or []
        self.scheduler = scheduler or TPUWorkloadScheduler()
        self.rewards: List[float] = []
        self.efficiency_history: List[Dict] = []
        self.optimization_count = 0
        self.best_reward = float('-inf')
        self.learning_rate = 0.05
        self.thermal_threshold = 85.0
        self._start_time = time.time()

    def monitor_nodes(self) -> List[Dict]:
        """Monitor all compute nodes and return status."""
        status = [node.get_status() for node in self.nodes]
        # Check for thermal alerts
        for s in status:
            if s['temp'] > self.thermal_threshold:
                logger.warning(f"THERMAL ALERT: Node {s['id']} at {s['temp']:.1f}C")
        return status

    def calculate_reward(self, status: List[Dict]) -> float:
        """
        Calculate reward using multi-objective optimization.
        Balances efficiency, thermal safety, and utilization.
        """
        reward = 0
        for s in status:
            if s['type'] == 'miner':
                efficiency = s['hash_rate'] / max(s['power'], 1)
                thermal_penalty = max(0, s['temp'] - self.thermal_threshold) * 0.5
                reward += efficiency * (100 - abs(s['temp'] - 70)) - thermal_penalty
            else:
                # TPU scoring: TFLOPS per watt, utilization bonus
                tpu_efficiency = s['tflops'] / max(s['power'], 1)
                util_bonus = s['utilization'] * 10
                reward += tpu_efficiency * 100 + util_bonus

        avg_reward = reward / max(len(status), 1)
        self.rewards.append(avg_reward)

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            logger.info(f"New best reward: {avg_reward:.4f}")

        return avg_reward

    def optimize_settings(self):
        """
        Apply adaptive optimization using reward gradient.
        Adjusts more aggressively when reward is improving, conservatively when stable.
        """
        self.optimization_count += 1

        # Calculate reward trend
        trend = 0.0
        if len(self.rewards) >= 3:
            recent = self.rewards[-3:]
            trend = recent[-1] - recent[0]

        # Adaptive optimization range based on trend
        if trend > 0:
            freq_range = (98, 103)
            volt_range = (97, 102)
        else:
            freq_range = (99, 101)
            volt_range = (99, 101)

        for node in self.nodes:
            # Thermal-aware optimization: reduce power if too hot
            if node.temp > self.thermal_threshold:
                freq = random.uniform(90, 95)
                volt = random.uniform(90, 95)
                logger.info(f"Cooling node {node.id}: reducing freq/volt")
            else:
                freq = random.uniform(*freq_range)
                volt = random.uniform(*volt_range)
            node.apply_optimization(freq, volt)
            node.uptime += 1.0

    def run_cycle(self, steps: int = 10, interval: float = 1.0):
        """Run optimization cycle with configurable steps."""
        logger.info(f"Starting optimization cycle: {steps} steps")
        for step in range(steps):
            status = self.monitor_nodes()
            reward = self.calculate_reward(status)
            self.efficiency_history.append({
                'step': step,
                'reward': reward,
                'avg_temp': sum(s['temp'] for s in status) / max(len(status), 1),
                'avg_power': sum(s['power'] for s in status) / max(len(status), 1),
                'timestamp': time.time(),
            })
            logger.info(f"Step {step+1}/{steps} - Reward: {reward:.4f}")
            self.optimize_settings()
            time.sleep(interval)

        logger.info(f"Cycle complete. Best reward: {self.best_reward:.4f}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        uptime = time.time() - self._start_time
        avg_reward = sum(self.rewards) / max(len(self.rewards), 1)
        compute = self.scheduler.get_total_compute()

        return {
            'version': self.VERSION,
            'uptime_seconds': round(uptime, 2),
            'nodes': len(self.nodes),
            'optimization_cycles': self.optimization_count,
            'avg_reward': round(avg_reward, 4),
            'best_reward': round(self.best_reward, 4),
            'reward_trend': self.rewards[-5:] if self.rewards else [],
            'total_compute': compute,
            'thermal_threshold': self.thermal_threshold,
        }


if __name__ == "__main__":
    # Initialize TPU scheduler with Brion Quantum TRC allocations
    scheduler = TPUWorkloadScheduler()
    compute = scheduler.get_total_compute()
    logger.info(f"Total compute: {compute['total_chips']} chips, {compute['total_pflops']:.1f} PFLOPS")

    # Create compute nodes (mix of TPUs and miners)
    nodes = []
    for i in range(8):
        nodes.append(ComputeNode(id=i, node_type='tpu', chip_type='v6e'))
    for i in range(8, 13):
        nodes.append(ComputeNode(id=i, node_type='miner'))

    # Run AI commander
    ai = ASICCommanderAI(nodes=nodes, scheduler=scheduler)

    # Submit example workloads
    scheduler.submit_job('quantum_simulation', 16, 'v6e')
    scheduler.submit_job('ml_training', 32, 'v5e')
    scheduler.submit_job('inference', 8, 'v4')

    ai.run_cycle(steps=10, interval=1.0)
    report = ai.get_performance_report()
    logger.info(f"Performance Report: {report}")
