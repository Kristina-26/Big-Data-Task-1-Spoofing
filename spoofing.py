import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import psutil
import csv
from collections import defaultdict
import threading


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) # degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0
    return R * c  # distance in km

def detect_anomalies(vessel_data):
    all_anomalies = []
    recent_positions = defaultdict(list)

    for mmsi, group in vessel_data:
        group = group.sort_values('# Timestamp').reset_index(drop=True)

        # Shifted columns for previous values
        group['prev_lat'] = group['Latitude'].shift(1)
        group['prev_lon'] = group['Longitude'].shift(1)
        group['prev_time'] = group['# Timestamp'].shift(1)
        group['prev_sog'] = group['SOG'].shift(1)
        group['prev_cog'] = group['COG'].shift(1)

        # Time difference in hours and seconds
        group['time_diff_h'] = (group['# Timestamp'] - group['prev_time']).dt.total_seconds() / 3600
        group['time_diff_s'] = (group['# Timestamp'] - group['prev_time']).dt.total_seconds()

        # Filtering out rows with 0 or missing time difference
        valid_mask = (group['time_diff_s'] > 0) & (~group['time_diff_s'].isna())
        group = group[valid_mask]

        # Distance between consecutive points
        group['distance_km'] = haversine(
            group['prev_lat'], group['prev_lon'], group['Latitude'], group['Longitude']
        )

        # Speed in knots
        group['calc_speed_knots'] = group['distance_km'] / group['time_diff_h'] / 1.852

        # A - Location Jumps
        jump_mask = (group['calc_speed_knots'] > 500)
        for t, speed in zip(group.loc[jump_mask, '# Timestamp'], group.loc[jump_mask, 'calc_speed_knots']):
            all_anomalies.append(("A - Location Jump", mmsi, None, t, speed))

        # B - Speed and Course Inconsistencies
        sog_diff = (group['SOG'] - group['prev_sog']).abs()
        cog_diff = np.minimum((group['COG'] - group['prev_cog']).abs() % 360,
                              360 - (group['COG'] - group['prev_cog']).abs() % 360)

        sog_rate = sog_diff / group['time_diff_s']
        cog_rate = cog_diff / group['time_diff_s']

        # B1 - Speed Acceleration
        speed_mask = (sog_rate > 2)
        for t, rate in zip(group.loc[speed_mask, '# Timestamp'], sog_rate[speed_mask]):
            all_anomalies.append(("B - Speed Acceleration", mmsi, None, t, rate))

        # B2 - Course Acceleration
        course_mask = (cog_rate > 90)
        for t, rate in zip(group.loc[course_mask, '# Timestamp'], cog_rate[course_mask]):
            all_anomalies.append(("B - Course Acceleration", mmsi, None, t, rate))

        # C - Conflicting Neighbors
        for i, row in group.iterrows():
            recent_positions[row['# Timestamp']].append((mmsi, row['Latitude'], row['Longitude'], row['# Timestamp']))

    # Vectorized C - Conflicting Neighbors
    seen_pairs = set()
    for time_key, positions in recent_positions.items():
        for i in range(len(positions)):
            mmsi1, lat1, lon1, t1 = positions[i]
            for j in range(i + 1, len(positions)):
                mmsi2, lat2, lon2, t2 = positions[j]
                if mmsi1 != mmsi2:
                    pair_key = tuple(sorted((mmsi1, mmsi2))) + (time_key,)
                    if pair_key in seen_pairs:
                        continue
                    dist = haversine(lat1, lon1, lat2, lon2)
                    if dist < 0.005:
                        all_anomalies.append(("C - Conflicting Neighbors", mmsi1, mmsi2, t1, dist))
                        seen_pairs.add(pair_key)

    return all_anomalies

def run_parallel_detection_with_monitoring(chunks, num_workers):
    with Pool(num_workers) as pool:
        # Starting monitoring
        monitor_thread, stop_event, cpu_data, mem_data = monitor_resources(pool, interval=0.5)
        monitor_thread.start()

        # Running the actual processing
        results = pool.map(detect_anomalies, chunks)

        # Stopping monitoring
        stop_event.set()
        monitor_thread.join(timeout=1.0)

    return results, cpu_data, mem_data

def benchmark_parallel(vessel_groups, max_workers=None):
    overall_start = time.time()

    if max_workers is None:
        max_workers = mp.cpu_count() - 1

    results = []
    best_parallel_result = []
    all_cpu_data = []
    all_memory_data = []

    # Sequential benchmark
    start_seq = time.time()
    seq_result = detect_anomalies(vessel_groups)
    end_seq = time.time()
    seq_time = end_seq - start_seq
    print(f"Sequential time: {seq_time:.2f} sec")

    for workers in range(2, max_workers + 1):
        batch_size = max(5, len(vessel_groups) // workers)
        chunks = batch_vessel_groups(vessel_groups, batch_size)

        # Process data with monitoring
        start = time.time()
        parallel_result, cpu_data, mem_data = run_parallel_detection_with_monitoring(chunks, workers)
        end = time.time()

        # Flatten the results
        parallel_result = [item for sublist in parallel_result for item in sublist]

        # Calculate metrics
        par_time = end - start
        speedup = seq_time / par_time if par_time > 0 else None

        # Calculate average resource usage from the collected data
        avg_worker_cpu = sum(data['workers'] for data in cpu_data) / len(cpu_data) if cpu_data else 0
        avg_worker_mem_percent = sum(data['workers_percent'] for data in mem_data) / len(mem_data) if mem_data else 0
        avg_worker_mem_gb = sum(data['workers_gb'] for data in mem_data) / len(mem_data) if mem_data else 0

        # Peak usage
        peak_worker_cpu = max(data['workers'] for data in cpu_data) if cpu_data else 0
        peak_worker_mem_percent = max(data['workers_percent'] for data in mem_data) if mem_data else 0
        peak_worker_mem_gb = max(data['workers_gb'] for data in mem_data) if mem_data else 0

        print(f"{workers} workers → {par_time:.2f} sec | speedup: {speedup:.2f}x | "
              f"Avg CPU: {avg_worker_cpu:.1f}% | Peak CPU: {peak_worker_cpu:.1f}% | "
              f"Avg Memory: {avg_worker_mem_percent:.2f}% ({avg_worker_mem_gb:.2f} GB) | "
              f"Peak Memory: {peak_worker_mem_percent:.2f}% ({peak_worker_mem_gb:.2f} GB)")

        results.append({
            "workers": workers,
            "parallel_time": par_time,
            "speedup": speedup,
            "avg_cpu_usage": avg_worker_cpu,
            "peak_cpu_usage": peak_worker_cpu,
            "avg_mem_usage": avg_worker_mem_percent,
            "peak_mem_usage": peak_worker_mem_percent,
            "avg_mem_usage_gb": avg_worker_mem_gb,
            "peak_mem_usage_gb": peak_worker_mem_gb
        })

        all_cpu_data.append(cpu_data)
        all_memory_data.append(mem_data)
        best_parallel_result = parallel_result

    overall_end = time.time()
    total_time = overall_end - overall_start
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")

    # Plot resource usage
    plot_resource_usage(results, all_cpu_data, all_memory_data)

    return seq_time, results, best_parallel_result

def batch_vessel_groups(groups, batch_size):
    # Splitting list of (mmsi, group) tuples into smaller batches
    return [groups[i:i + batch_size] for i in range(0, len(groups), batch_size)]

def plot_benchmark_results(seq_time, results):
    workers = [r['workers'] for r in results]
    times = [r['parallel_time'] for r in results]
    speedups = [r['speedup'] for r in results]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Execution Time (sec)", color='tab:red')
    ax1.plot(workers, times, marker='o', color='tab:red', label="Parallel Time")
    ax1.axhline(y=seq_time, color='tab:gray', linestyle='--', label="Sequential Time")
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Speedup", color='tab:blue')
    ax2.plot(workers, speedups, marker='x', color='tab:blue', label="Speedup")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title("Parallel Processing Benchmark")
    plt.show()


def plot_resource_usage(benchmark_results, cpu_data, memory_data):
    # Plotting improved CPU and memory usage for different worker counts
    workers = [r['workers'] for r in benchmark_results]
    times = [r['parallel_time'] for r in benchmark_results]
    speedups = [r['speedup'] for r in benchmark_results]
    avg_cpu = [r['avg_cpu_usage'] for r in benchmark_results]
    peak_cpu = [r['peak_cpu_usage'] for r in benchmark_results]
    avg_mem_percent = [r['avg_mem_usage'] for r in benchmark_results]
    peak_mem_percent = [r['peak_mem_usage'] for r in benchmark_results]

    # Calculate memory usage in GB
    avg_mem_gb = [r.get('avg_mem_usage_gb', 0) for r in benchmark_results]
    peak_mem_gb = [r.get('peak_mem_usage_gb', 0) for r in benchmark_results]

    # Create a figure with 4 subplots (added one for GB memory)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    # Plot 1: Execution Time and Speedup
    ax1.set_ylabel("Execution Time (sec)", color='tab:red')
    ax1.plot(workers, times, marker='o', color='tab:red', label="Parallel Time")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Speedup", color='tab:blue')
    ax1_twin.plot(workers, speedups, marker='x', color='tab:blue', label="Speedup")
    ax1_twin.tick_params(axis='y', labelcolor='tab:blue')

    ax1.set_title("Performance Metrics by Worker Count")
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: CPU Usage
    ax2.set_ylabel("CPU Usage (%)")
    ax2.plot(workers, avg_cpu, marker='s', color='tab:green', label="Average CPU")
    ax2.plot(workers, peak_cpu, marker='*', color='darkgreen', label="Peak CPU")
    ax2.set_title("CPU Usage by Worker Count")
    ax2.legend()

    # Plot 3: Memory Usage (Percent)
    ax3.set_ylabel("Memory Usage (%)")
    ax3.plot(workers, avg_mem_percent, marker='d', color='tab:purple', label="Average Memory")
    ax3.plot(workers, peak_mem_percent, marker='*', color='darkviolet', label="Peak Memory")
    ax3.set_title("Memory Usage by Worker Count (Percentage)")
    ax3.legend()

    # Plot 4: Memory Usage (GB)
    ax4.set_ylabel("Memory Usage (GB)")
    ax4.plot(workers, avg_mem_gb, marker='d', color='tab:orange', label="Average Memory")
    ax4.plot(workers, peak_mem_gb, marker='*', color='darkorange', label="Peak Memory")
    ax4.set_title("Memory Usage by Worker Count (GB)")
    ax4.set_xlabel("Number of Workers")
    ax4.legend()

    plt.tight_layout()
    plt.savefig('resource_usage_analysis.png')
    plt.show()

def monitor_resources(worker_pool, interval=0.5):
    # Monitoring CPU and memory usage of specific worker processes
    cpu_percentages = []
    memory_usages = []
    stop_monitoring = threading.Event()

    # Getting process IDs from the worker pool
    worker_pids = [process.pid for process in worker_pool._pool]

    def resource_monitor():
        while not stop_monitoring.is_set():
            # System-wide metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()
            system_mem_percent = system_memory.percent
            system_mem_gb = system_memory.used / (1024**3)  # Convert bytes to GB

            # Worker-specific metrics
            worker_cpu = 0
            worker_mem_percent = 0
            worker_mem_gb = 0
            for pid in worker_pids:
                try:
                    process = psutil.Process(pid)
                    with process.oneshot():  # Getting all process info in one call
                        worker_cpu += process.cpu_percent(interval=0.1)
                        worker_mem_percent += process.memory_percent()
                        worker_mem_gb += process.memory_info().rss / (1024**3)  # Convert bytes to GB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might have ended
                    pass

            cpu_percentages.append({
                'system': system_cpu,
                'workers': worker_cpu
            })
            memory_usages.append({
                'system_percent': system_mem_percent,
                'workers_percent': worker_mem_percent,
                'system_gb': system_mem_gb,
                'workers_gb': worker_mem_gb
            })

            time.sleep(interval)

    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.daemon = True
    return monitor_thread, stop_monitoring, cpu_percentages, memory_usages

def save_anomalies_with_type(anomalies, output_file='spoofing_anomalies_labeled.csv'):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Anomaly Type', 'MMSI_1', 'MMSI_2 (if applicable)', 'Timestamp', 'Speed/Diff/Distance'])

        for entry in anomalies:
            anomaly_type, mmsi1, mmsi2, timestamp, value = entry
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, pd.Timestamp) else str(timestamp)
            writer.writerow([
                anomaly_type,
                mmsi1,
                mmsi2 if mmsi2 is not None else '',
                ts_str,
                round(value, 3)
            ])

if __name__ == '__main__':
    df = pd.read_csv('aisdk-2025-01-01.csv')

    # Optimizing dtypes
    df['MMSI'] = df['MMSI'].astype('int32')
    df['Latitude'] = df['Latitude'].astype('float32')
    df['Longitude'] = df['Longitude'].astype('float32')
    df['SOG'] = df['SOG'].astype('float32')
    df['COG'] = df['COG'].astype('float32')

    # Parsing timestamp just once
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])

    df.columns = df.columns.str.strip()

    # Removing rows with missing values in key columns
    df = df.dropna(subset=["# Timestamp", "MMSI", "Longitude", "Latitude", "SOG", "COG"])
    num_vessels = df['MMSI'].nunique()
    print(f"Number of vessels: {num_vessels}")
    vessel_groups = list(df.groupby('MMSI'))

    seq_time, benchmark_results, final_anomalies = benchmark_parallel(vessel_groups)
    plot_benchmark_results(seq_time, benchmark_results)

    print(f"\nTotal anomalies detected: {len(final_anomalies)}\n")
    for anomaly in final_anomalies[:10]:
        print(anomaly)

    spoofing_mmsis = set()
    for anomaly in final_anomalies:
        _, mmsi1, mmsi2, _, _ = anomaly
        spoofing_mmsis.add(mmsi1)
        if mmsi2 is not None:
            spoofing_mmsis.add(mmsi2)

    print(f"\nNumber of vessels that showed spoofing behavior: {len(spoofing_mmsis)}")

    spoofing_percent = len(spoofing_mmsis) / num_vessels * 100
    print(f"Spoofing percentage: {spoofing_percent:.2f}%")

    # Suspicion counts per vessel
    suspicion_score = defaultdict(int)

    for anomaly in final_anomalies:
        _, mmsi1, mmsi2, _, _ = anomaly
        suspicion_score[mmsi1] += 1
        if mmsi2 is not None:
            suspicion_score[mmsi2] += 1

    # Most suspicious vessels
    min_anomalies_threshold = 5
    spoofing_mmsis = {mmsi for mmsi, count in suspicion_score.items() if count >= min_anomalies_threshold}

    print(f"\nNumber of vessels that showed spoofing behavior (with at least {min_anomalies_threshold} anomalies): {len(spoofing_mmsis)}")

    spoofing_percent = len(spoofing_mmsis) / num_vessels * 100
    print(f"Spoofing percentage: {spoofing_percent:.2f}%")

    sorted_suspects = sorted(suspicion_score.items(), key=lambda x: x[1], reverse=True)

    print("\nTop Suspicious Vessels (by number of anomalies):")
    for mmsi, count in sorted_suspects[:10]:
        spoofing_status = "SPOOFING" if count >= min_anomalies_threshold else "Low Confidence"
        print(f"MMSI: {mmsi}, Number of anomalies: {count}, Status: {spoofing_status}")

    # Saving spoofing anomalies to CSV
    save_anomalies_with_type(final_anomalies)
    print("Labeled spoofing anomalies saved to 'spoofing_anomalies_labeled.csv'")