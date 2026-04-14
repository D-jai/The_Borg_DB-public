# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Health Monitoring Package
=========================

This package provides health monitoring and metrics collection
for the ctxmtg system. It tracks resource usage (RAM, CPU, disk),
database statistics (record counts, index sizes), and processing
metrics (ingestion throughput, query latency).

Health data is exposed via:
    - CLI command: `ctxmtg health` (human-readable output)
    - JSONL log file: machine-readable metrics for dashboards
    - Query server endpoint: /health (when running as a service)

This is especially important for edge deployments (Raspberry Pi)
where resource constraints are tight and monitoring helps prevent
out-of-memory crashes or disk-full conditions.

Submodules:
    - monitor.py : Health endpoint, metrics JSONL log
"""
