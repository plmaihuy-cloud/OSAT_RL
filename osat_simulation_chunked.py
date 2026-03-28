#!/usr/bin/env python3
"""
OSAT Thesis – Chunked Simulation Runner (Auto‑creates input CSV files)
Usage examples:
    python osat_simulation_chunked.py --start 0 --end 26 --replications 10 --duration 168 --output chunk_0.csv
    python osat_simulation_chunked.py --policy FIFO --replications 10
    python osat_simulation_chunked.py --start 0 --end 5 --replications 1 --duration 1   # quick test
"""

import os
import argparse
import numpy as np
import pandas as pd
import simpy
import warnings
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 0. Ensure input data exists (create default CSV files if missing)
# ------------------------------------------------------------
def ensure_input_files(data_folder='input_data'):
    """Create all required input CSV files with default content if they don't exist."""
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Stage configuration
    stage_path = os.path.join(data_folder, 'stage_config_base.csv')
    if not os.path.exists(stage_path):
        with open(stage_path, 'w') as f:
            f.write("stage_id,name,num_machines,base_time_min,cv,mtbf_hours,mttr_hours\n")
            f.write("0,DieAttach,3,3.5,0.2,480,4\n")
            f.write("1,WireBond,4,4.2,0.25,360,5\n")
            f.write("2,Encapsulation,3,6.8,0.22,600,6\n")
            f.write("3,FinalTest,5,2.1,0.3,500,3\n")
            f.write("4,Quality,2,0.8,0.15,1000,2\n")

    # Product mix
    prod_path = os.path.join(data_folder, 'product_mix.csv')
    if not os.path.exists(prod_path):
        with open(prod_path, 'w') as f:
            f.write("product_type,probability\nA,0.5\nB,0.3\nC,0.2\n")

    # Global parameters
    glob_path = os.path.join(data_folder, 'global_params.csv')
    if not os.path.exists(glob_path):
        with open(glob_path, 'w') as f:
            f.write("param,value\nplanned_lead_time_hours,168\nsimulation_duration_hours,168\n")

    # Variability parameters
    var_path = os.path.join(data_folder, 'variability_parameters.csv')
    if not os.path.exists(var_path):
        with open(var_path, 'w') as f:
            f.write("variability,cv_multiplier,mtbf_multiplier,arrival_rate_multiplier\n")
            f.write("Low,0.5,1.5,1.0\nMedium,1.0,1.0,1.0\nHigh,1.5,0.67,1.0\n")

    # Failure parameters
    fail_path = os.path.join(data_folder, 'failure_parameters.csv')
    if not os.path.exists(fail_path):
        with open(fail_path, 'w') as f:
            f.write("failure,mtbf_multiplier\nLow,1.5\nMedium,1.0\nHigh,0.67\n")

    # Demand patterns (parameters column may contain commas)
    demand_path = os.path.join(data_folder, 'demand_patterns.csv')
    if not os.path.exists(demand_path):
        with open(demand_path, 'w') as f:
            f.write("demand,type,description,parameters\n")
            f.write("Steady,constant,Constant arrival rate,1.0\n")
            f.write("Seasonal,sinusoidal,Weekly cycle (amplitude 0.3),0.3,168\n")
            f.write("Surge,step,2-day surge (factor 1.5),1.5,72,120\n")

    # Full condition list (7 policies × 27 combos = 189 rows)
    cond_path = os.path.join(data_folder, 'experiment_conditions_full.csv')
    if not os.path.exists(cond_path):
        policies = ['FIFO', 'SPT', 'EDD', 'H-MARL-CTDE', 'H-MARL-Independent',
                    'H-MARL-ValueDecomp', 'Flat MARL']
        variability = ['Low', 'Medium', 'High']
        demand = ['Steady', 'Seasonal', 'Surge']
        failure = ['Low', 'Medium', 'High']
        rows = []
        cond_id = 0
        for pol in policies:
            for var in variability:
                for dem in demand:
                    for fail in failure:
                        rows.append([cond_id, pol, var, dem, fail])
                        cond_id += 1
        df_cond = pd.DataFrame(rows, columns=['condition_id', 'policy', 'variability', 'demand', 'failure'])
        df_cond.to_csv(cond_path, index=False)

    print("All input CSV files are present (created if missing).")

# ------------------------------------------------------------
# 1. Load input data (auto‑created if missing)
# ------------------------------------------------------------
data_folder = 'input_data'
ensure_input_files(data_folder)

stages_df = pd.read_csv(os.path.join(data_folder, 'stage_config_base.csv'))
product_mix_df = pd.read_csv(os.path.join(data_folder, 'product_mix.csv'))
global_df = pd.read_csv(os.path.join(data_folder, 'global_params.csv'))
var_df = pd.read_csv(os.path.join(data_folder, 'variability_parameters.csv'))
fail_df = pd.read_csv(os.path.join(data_folder, 'failure_parameters.csv'))

# Demand patterns (handle commas in parameters column)
with open(os.path.join(data_folder, 'demand_patterns.csv'), 'r') as f:
    lines = f.readlines()
header = lines[0].strip().split(',')
data = []
for line in lines[1:]:
    parts = line.strip().split(',')
    demand = parts[0]
    typ = parts[1]
    description = parts[2]
    parameters = ','.join(parts[3:])
    data.append([demand, typ, description, parameters])
demand_df = pd.DataFrame(data, columns=header)

conditions_df = pd.read_csv(os.path.join(data_folder, 'experiment_conditions_full.csv'))

product_types = product_mix_df['product_type'].tolist()
product_probs = product_mix_df['probability'].tolist()
global_dict = global_df.set_index('param')['value'].to_dict()
sim_duration = float(global_dict['simulation_duration_hours'])
planned_lead_time = float(global_dict['planned_lead_time_hours'])
var_dict = var_df.set_index('variability').to_dict('index')
fail_dict = fail_df.set_index('failure').to_dict('index')
demand_dict = demand_df.set_index('demand').to_dict('index')

# ------------------------------------------------------------
# 2. Simulation classes (identical to original)
# ------------------------------------------------------------
class Job:
    def __init__(self, job_id, product_type, arrival_time, due_date, quantity=100):
        self.job_id = job_id
        self.product_type = product_type
        self.arrival_time = arrival_time
        self.due_date = due_date
        self.quantity = quantity
        self.completion_time = None
        self.stage_times = []

class Machine:
    def __init__(self, env, machine_id, stage, config):
        self.env = env
        self.machine_id = machine_id
        self.stage = stage
        self.base_time = config['base_time_min'] / 60.0
        self.cv = config['cv']
        self.status = 'idle'
        self.current_job = None
        self.total_busy_time = 0
        self.last_busy_start = None
        self.mtbf = config['mtbf_hours']
        self.mttr = config['mttr_hours']
        self.failure_process = self.env.process(self.run_failures())

    def run_failures(self):
        while True:
            ttf = np.random.exponential(self.mtbf)
            yield self.env.timeout(ttf)
            if self.status == 'busy':
                if self.current_job:
                    self.current_job.process.interrupt()
                self.status = 'down'
                yield self.env.timeout(np.random.exponential(self.mttr))
                self.status = 'idle'
            else:
                yield self.env.timeout(1)

    def process(self, job, duration):
        self.status = 'busy'
        self.current_job = job
        self.last_busy_start = self.env.now
        try:
            yield self.env.timeout(duration)
        except simpy.Interrupt:
            pass
        finally:
            self.total_busy_time += self.env.now - self.last_busy_start
            self.status = 'idle'
            self.current_job = None

    def get_processing_time(self, product_type):
        mu = np.log(self.base_time) - 0.5 * np.log(1 + self.cv**2)
        sigma = np.sqrt(np.log(1 + self.cv**2))
        return np.random.lognormal(mu, sigma)

class OSATSimulation:
    def __init__(self, stage_configs, job_arrivals, dispatching_rule='FIFO'):
        self.env = simpy.Environment()
        self.stage_configs = stage_configs
        self.job_arrivals = job_arrivals
        self.dispatching_rule = dispatching_rule
        self.stages = []
        self.machines = {}
        self.queues = {}
        self.available_machines = {}
        self.completed_jobs = []
        self.metrics = {}
        self._setup_stages()

    def _setup_stages(self):
        for cfg in self.stage_configs:
            stage_id = cfg['stage_id']
            machines = [Machine(self.env, f"{cfg['name']}_{i}", stage_id, cfg) for i in range(cfg['num_machines'])]
            self.machines[stage_id] = machines
            self.queues[stage_id] = []
            self.available_machines[stage_id] = simpy.Store(self.env)
            for m in machines:
                self.available_machines[stage_id].put(m)
            self.stages.append({
                'id': stage_id,
                'name': cfg['name'],
                'machines': machines,
                'queue': self.queues[stage_id],
                'available': self.available_machines[stage_id]
            })
        for stage in self.stages:
            self.env.process(self._dispatch(stage))

    def _dispatch(self, stage):
        while True:
            machine = yield stage['available'].get()
            if stage['queue']:
                if self.dispatching_rule == 'SPT':
                    stage['queue'].sort(key=lambda job: self._base_processing_time(job.product_type))
                elif self.dispatching_rule == 'EDD':
                    stage['queue'].sort(key=lambda job: job.due_date)
                job = stage['queue'].pop(0)
                self.env.process(self._process_job_on_machine(job, machine, stage['id']))
            else:
                stage['available'].put(machine)

    def _base_processing_time(self, product_type):
        base_times = {'A': 3.5, 'B': 3.8, 'C': 4.1}
        return base_times.get(product_type, 3.5) / 60.0

    def _process_job_on_machine(self, job, machine, stage_id):
        job.stage_times.append((stage_id, self.env.now, None))
        duration = machine.get_processing_time(job.product_type)
        try:
            yield self.env.process(machine.process(job, duration))
        except simpy.Interrupt:
            pass
        for idx, (sid, entry, _) in enumerate(job.stage_times):
            if sid == stage_id and job.stage_times[idx][2] is None:
                job.stage_times[idx] = (sid, entry, self.env.now)
                break
        self.stages[stage_id]['available'].put(machine)
        if stage_id == len(self.stages) - 1:
            job.completion_time = self.env.now
            self.completed_jobs.append(job)

    def run(self, duration_hours):
        self.env.process(self._job_generator())
        self.env.run(until=duration_hours)
        if self.completed_jobs:
            self.metrics['makespan'] = max(j.completion_time for j in self.completed_jobs)
            self.metrics['flow_times'] = [j.completion_time - j.arrival_time for j in self.completed_jobs]
            self.metrics['throughput'] = len(self.completed_jobs) / duration_hours
            self.metrics['total_units'] = sum(j.quantity for j in self.completed_jobs)
            total_idle = sum(m.total_busy_time for mlist in self.machines.values() for m in mlist)
            self.metrics['energy'] = self.metrics['total_units'] * 0.3 + 2.5 * total_idle / 3600
        else:
            self.metrics = {'makespan': 0, 'flow_times': [], 'throughput': 0, 'total_units': 0, 'energy': 0}
        return self.metrics

    def _job_generator(self):
        for job_data in sorted(self.job_arrivals, key=lambda x: x['arrival_time']):
            yield self.env.timeout(job_data['arrival_time'] - self.env.now)
            job = Job(
                job_id=len(self.completed_jobs) + len(self.job_arrivals),
                product_type=job_data['product_type'],
                arrival_time=self.env.now,
                due_date=job_data['due_date'],
                quantity=job_data['quantity']
            )
            self.queues[0].append(job)

# ------------------------------------------------------------
# 3. Helper to generate job arrivals
# ------------------------------------------------------------
def generate_jobs_for_condition(arrival_rate_base, demand_pattern, duration_hours,
                                product_list, product_probs, planned_lead_time):
    pattern_type = demand_pattern['type']
    if pattern_type == 'constant':
        def arrival_rate(t):
            return arrival_rate_base
    elif pattern_type == 'sinusoidal':
        params = demand_pattern['parameters'].split(',')
        amplitude = float(params[0])
        period = float(params[1])
        def arrival_rate(t):
            return arrival_rate_base * (1 + amplitude * np.sin(2 * np.pi * t / period))
    elif pattern_type == 'step':
        params = demand_pattern['parameters'].split(',')
        factor = float(params[0])
        start = float(params[1])
        end = float(params[2])
        def arrival_rate(t):
            if start <= t <= end:
                return arrival_rate_base * factor
            else:
                return arrival_rate_base
    else:
        raise ValueError(f"Unknown demand pattern type: {pattern_type}")

    jobs = []
    t = 0.0
    job_id = 0
    while t < duration_hours:
        rate = arrival_rate(t)
        if rate <= 0:
            t += 1.0
            continue
        interarrival = np.random.exponential(1.0 / rate)
        t += interarrival
        if t >= duration_hours:
            break
        product_type = np.random.choice(product_list, p=product_probs)
        due_date = t + planned_lead_time
        jobs.append({
            'job_id': job_id,
            'arrival_time': t,
            'product_type': product_type,
            'due_date': due_date,
            'quantity': 100
        })
        job_id += 1
    return jobs

# ------------------------------------------------------------
# 4. Main chunk runner
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run a chunk of OSAT simulation conditions.')
    parser.add_argument('--start', type=int, help='Starting condition_id (inclusive)')
    parser.add_argument('--end', type=int, help='Ending condition_id (inclusive)')
    parser.add_argument('--policy', type=str, help='Filter by policy (e.g., FIFO, SPT, EDD)')
    parser.add_argument('--variability', type=str, help='Filter by variability (Low, Medium, High)')
    parser.add_argument('--demand', type=str, help='Filter by demand (Steady, Seasonal, Surge)')
    parser.add_argument('--failure', type=str, help='Filter by failure (Low, Medium, High)')
    parser.add_argument('--replications', type=int, default=10, help='Number of replications per condition')
    parser.add_argument('--duration', type=float, default=None, help='Override simulation duration (hours)')
    parser.add_argument('--output', type=str, default='heuristic_results_chunk.csv', help='Output CSV filename')
    args = parser.parse_args()

    # Build condition filter
    mask = pd.Series(True, index=conditions_df.index)
    if args.start is not None:
        mask &= conditions_df['condition_id'] >= args.start
    if args.end is not None:
        mask &= conditions_df['condition_id'] <= args.end
    if args.policy is not None:
        mask &= conditions_df['policy'] == args.policy
    if args.variability is not None:
        mask &= conditions_df['variability'] == args.variability
    if args.demand is not None:
        mask &= conditions_df['demand'] == args.demand
    if args.failure is not None:
        mask &= conditions_df['failure'] == args.failure

    selected_conditions = conditions_df[mask]
    if selected_conditions.empty:
        print("No conditions match the given filters.")
        return

    print(f"Running {len(selected_conditions)} conditions with {args.replications} replications each.")

    all_results = []
    for idx, row in selected_conditions.iterrows():
        print(f"Condition {row['condition_id']}: {row['policy']} {row['variability']} {row['demand']} {row['failure']}")

        var_mult = var_dict[row['variability']]
        fail_mult = fail_dict[row['failure']]['mtbf_multiplier']
        demand_info = demand_dict[row['demand']]
        base_arrival_rate = 0.5 * var_mult['arrival_rate_multiplier']

        # Build stage configs with multipliers
        stage_cfgs = []
        for _, s in stages_df.iterrows():
            cfg = s.to_dict()
            cfg['cv'] = s['cv'] * var_mult['cv_multiplier']
            cfg['mtbf_hours'] = s['mtbf_hours'] * var_mult['mtbf_multiplier'] * fail_mult
            stage_cfgs.append(cfg)

        # Use custom duration if provided
        duration = args.duration if args.duration is not None else sim_duration
        jobs = generate_jobs_for_condition(
            arrival_rate_base=base_arrival_rate,
            demand_pattern=demand_info,
            duration_hours=duration,
            product_list=product_types,
            product_probs=product_probs,
            planned_lead_time=planned_lead_time
        )

        rep_metrics = []
        for rep in range(args.replications):
            sim = OSATSimulation(stage_cfgs, jobs, dispatching_rule=row['policy'])
            metrics = sim.run(duration)
            rep_metrics.append({
                'makespan': metrics['makespan'],
                'flow_time_mean': np.mean(metrics['flow_times']) if metrics['flow_times'] else 0,
                'throughput': metrics['throughput'],
                'energy_per_unit': metrics['energy'] / metrics['total_units'] if metrics['total_units'] else 0,
                'total_units': metrics['total_units']
            })

        all_results.append({
            'condition_id': row['condition_id'],
            'policy': row['policy'],
            'variability': row['variability'],
            'demand': row['demand'],
            'failure': row['failure'],
            'makespan_mean': np.mean([m['makespan'] for m in rep_metrics]),
            'makespan_std': np.std([m['makespan'] for m in rep_metrics]),
            'flow_time_mean': np.mean([m['flow_time_mean'] for m in rep_metrics]),
            'flow_time_std': np.std([m['flow_time_mean'] for m in rep_metrics]),
            'throughput_mean': np.mean([m['throughput'] for m in rep_metrics]),
            'throughput_std': np.std([m['throughput'] for m in rep_metrics]),
            'energy_mean': np.mean([m['energy_per_unit'] for m in rep_metrics]),
            'energy_std': np.std([m['energy_per_unit'] for m in rep_metrics])
        })

    chunk_df = pd.DataFrame(all_results)
    chunk_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
