#!/usr/bin/env python3
"""
Report Cleanup Script for ViktorAI.

This script helps manage the report files from evaluator tests and benchmarks
by cleaning up older reports while keeping a configurable number of the most recent ones.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

def find_reports(base_dir, report_type):
    """
    Find all report directories of a specific type.
    
    Args:
        base_dir: Base directory to search
        report_type: Type of reports to find ('benchmark' or 'evaluator')
        
    Returns:
        List of tuples (path, timestamp) for each run directory found
    """
    if not os.path.exists(base_dir):
        return []
    
    base_path = Path(base_dir)
    results = []
    
    # Look for run_YYYYMMDD_HHMMSS directories
    run_pattern = re.compile(r'run_(\d{8}_\d{6})')
    
    for model_family in base_path.iterdir():
        if not model_family.is_dir():
            continue
            
        for model_dir in model_family.iterdir():
            if not model_dir.is_dir():
                continue
                
            for run_dir in model_dir.iterdir():
                match = run_pattern.match(run_dir.name)
                if match and run_dir.is_dir():
                    timestamp_str = match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        results.append((run_dir, timestamp))
                    except ValueError:
                        # If the timestamp isn't in the expected format, skip it
                        continue
    
    return sorted(results, key=lambda x: x[1])

def cleanup_by_age(reports, max_age_days):
    """
    Clean up reports older than the specified age.
    
    Args:
        reports: List of (path, timestamp) tuples for all reports
        max_age_days: Maximum age in days to keep
        
    Returns:
        List of paths that were deleted
    """
    if max_age_days <= 0:
        return []
        
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    deleted = []
    
    for report_path, timestamp in reports:
        if timestamp < cutoff_date:
            shutil.rmtree(report_path)
            deleted.append(report_path)
    
    return deleted

def cleanup_by_count(reports, keep_count):
    """
    Keep only the specified number of most recent reports per model.
    
    Args:
        reports: List of (path, timestamp) tuples for all reports
        keep_count: Number of most recent reports to keep per model
        
    Returns:
        List of paths that were deleted
    """
    if keep_count <= 0 or len(reports) <= keep_count:
        return []
    
    # Group reports by model
    reports_by_model = {}
    for report_path, timestamp in reports:
        model_path = report_path.parent
        if model_path not in reports_by_model:
            reports_by_model[model_path] = []
        reports_by_model[model_path].append((report_path, timestamp))
    
    # Sort reports by timestamp
    for model_path in reports_by_model:
        reports_by_model[model_path].sort(key=lambda x: x[1])
    
    # Delete older reports for each model
    deleted = []
    for model_path, model_reports in reports_by_model.items():
        if len(model_reports) > keep_count:
            to_delete = model_reports[:-keep_count]
            for report_path, _ in to_delete:
                shutil.rmtree(report_path)
                deleted.append(report_path)
    
    return deleted

def main():
    """Main function for the report cleanup script."""
    parser = argparse.ArgumentParser(description="Clean up old benchmark and evaluator test reports")
    parser.add_argument("--benchmark-dir", type=str, default="tests/benchmark_results",
                        help="Directory containing benchmark reports")
    parser.add_argument("--evaluator-dir", type=str, default="tests/evaluator_test_results",
                        help="Directory containing evaluator test reports")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Maximum age in days to keep reports (0 to disable)")
    parser.add_argument("--keep-count", type=int, default=5,
                        help="Number of most recent reports to keep per model (0 to disable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--report-type", choices=["benchmark", "evaluator", "both"], default="both",
                        help="Type of reports to clean up")
    
    args = parser.parse_args()
    
    # Find reports
    benchmark_reports = []
    evaluator_reports = []
    
    if args.report_type in ["benchmark", "both"]:
        benchmark_reports = find_reports(args.benchmark_dir, "benchmark")
        print(f"Found {len(benchmark_reports)} benchmark reports")
        
    if args.report_type in ["evaluator", "both"]:
        evaluator_reports = find_reports(args.evaluator_dir, "evaluator")
        print(f"Found {len(evaluator_reports)} evaluator test reports")
    
    # Clean up by age
    if args.max_age > 0:
        if args.report_type in ["benchmark", "both"]:
            deleted = cleanup_by_age(benchmark_reports, args.max_age) if not args.dry_run else []
            print(f"Would delete {len([r for r, t in benchmark_reports if t < datetime.now() - timedelta(days=args.max_age)])} benchmark reports older than {args.max_age} days" if args.dry_run else f"Deleted {len(deleted)} benchmark reports older than {args.max_age} days")
            
        if args.report_type in ["evaluator", "both"]:
            deleted = cleanup_by_age(evaluator_reports, args.max_age) if not args.dry_run else []
            print(f"Would delete {len([r for r, t in evaluator_reports if t < datetime.now() - timedelta(days=args.max_age)])} evaluator test reports older than {args.max_age} days" if args.dry_run else f"Deleted {len(deleted)} evaluator test reports older than {args.max_age} days")
    
    # Clean up by count
    if args.keep_count > 0:
        if args.report_type in ["benchmark", "both"]:
            deleted = cleanup_by_count(benchmark_reports, args.keep_count) if not args.dry_run else []
            if args.dry_run:
                # Count how many would be deleted
                reports_by_model = {}
                for report_path, timestamp in benchmark_reports:
                    model_path = report_path.parent
                    if model_path not in reports_by_model:
                        reports_by_model[model_path] = []
                    reports_by_model[model_path].append((report_path, timestamp))
                
                would_delete = sum(max(0, len(reports) - args.keep_count) for reports in reports_by_model.values())
                print(f"Would delete {would_delete} older benchmark reports, keeping {args.keep_count} most recent per model")
            else:
                print(f"Deleted {len(deleted)} older benchmark reports, keeping {args.keep_count} most recent per model")
            
        if args.report_type in ["evaluator", "both"]:
            deleted = cleanup_by_count(evaluator_reports, args.keep_count) if not args.dry_run else []
            if args.dry_run:
                # Count how many would be deleted
                reports_by_model = {}
                for report_path, timestamp in evaluator_reports:
                    model_path = report_path.parent
                    if model_path not in reports_by_model:
                        reports_by_model[model_path] = []
                    reports_by_model[model_path].append((report_path, timestamp))
                
                would_delete = sum(max(0, len(reports) - args.keep_count) for reports in reports_by_model.values())
                print(f"Would delete {would_delete} older evaluator test reports, keeping {args.keep_count} most recent per model")
            else:
                print(f"Deleted {len(deleted)} older evaluator test reports, keeping {args.keep_count} most recent per model")
    
    print("Done")

if __name__ == "__main__":
    main() 