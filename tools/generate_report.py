"""
Generate markdown comparison report from policy evaluation results.

Converts comparison JSON (from compare_policies.py) to a formatted markdown
report with summary tables, percent improvements, and statistical tests.

Usage:
    python tools/generate_report.py \
        --results comparison_results.json \
        --baseline random \
        --out comparison_report.md
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
from datetime import datetime


def load_results(json_path):
    """Load comparison results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_improvement(baseline_val, policy_val, lower_is_better=True):
    """
    Compute percent improvement vs baseline.

    Args:
        baseline_val: Baseline metric value
        policy_val: Policy metric value
        lower_is_better: If True, reduction is improvement (queue, wait)
                        If False, increase is improvement (reward)

    Returns:
        Percent improvement (positive = better)
    """
    if abs(baseline_val) < 1e-9:
        return 0.0

    if lower_is_better:
        # Lower is better (queue, wait): (baseline - policy) / baseline * 100
        return ((baseline_val - policy_val) / abs(baseline_val)) * 100
    else:
        # Higher is better (reward): (policy - baseline) / |baseline| * 100
        return ((policy_val - baseline_val) / abs(baseline_val)) * 100


def statistical_test(baseline_samples, policy_samples):
    """
    Run t-test and Mann-Whitney U test for statistical significance.

    Returns:
        dict with p-values and significance flag
    """
    try:
        from scipy import stats
    except ImportError:
        return {"error": "scipy not available"}

    baseline_samples = np.array(baseline_samples)
    policy_samples = np.array(policy_samples)

    # Two-sample t-test
    t_stat, t_pval = stats.ttest_ind(baseline_samples, policy_samples)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(baseline_samples, policy_samples, alternative='two-sided')

    return {
        "t_pval": float(t_pval),
        "u_pval": float(u_pval),
        "significant": (t_pval < 0.05)
    }


def generate_markdown_report(results, baseline_name, output_path):
    """Generate markdown report."""

    if baseline_name not in results:
        print(f"⚠ Baseline '{baseline_name}' not found in results")
        print(f"  Available policies: {list(results.keys())}")
        return False

    baseline = results[baseline_name]

    # Start report
    md = []
    md.append("# Traffic Signal Control Policy Comparison Report\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Baseline:** {baseline_name}\n")
    md.append(f"**Policies Evaluated:** {', '.join(results.keys())}\n")
    md.append("\n---\n")

    # Summary Table
    md.append("## Summary Statistics\n")
    md.append("| Policy | Reward | Queue (mean) | Queue (p95) | Wait (mean) | Wait (p95) | Long Waiters |\n")
    md.append("|--------|--------|--------------|-------------|-------------|------------|-------------|\n")

    for policy_name, policy_data in results.items():
        s = policy_data["summary"]
        md.append(f"| {policy_name:10s} | {s['reward_mean']:7.2f} | "
                  f"{s['queue_mean']:6.2f} | {s['queue_p95']:6.2f} | "
                  f"{s['wait_mean']:6.2f} | {s['wait_p95']:6.2f} | "
                  f"{s['long_wait_mean']:6.2f} |\n")

    md.append("\n---\n")

    # Percent Improvement Table
    md.append("## Percent Improvement vs Baseline\n")
    md.append(f"*Baseline: {baseline_name}*\n\n")
    md.append("| Policy | Queue Reduction | Wait Reduction | Long Wait Reduction | Reward Improvement |\n")
    md.append("|--------|-----------------|----------------|--------------------|--------------------|\n")

    baseline_summary = baseline["summary"]

    for policy_name, policy_data in results.items():
        if policy_name == baseline_name:
            md.append(f"| {policy_name:10s} | — | — | — | — |\n")
            continue

        s = policy_data["summary"]

        queue_imp = compute_improvement(baseline_summary["queue_mean"], s["queue_mean"], lower_is_better=True)
        wait_imp = compute_improvement(baseline_summary["wait_mean"], s["wait_mean"], lower_is_better=True)
        long_wait_imp = compute_improvement(baseline_summary["long_wait_mean"], s["long_wait_mean"], lower_is_better=True)
        reward_imp = compute_improvement(baseline_summary["reward_mean"], s["reward_mean"], lower_is_better=False)

        # Format with color indicators
        def format_pct(val):
            if val > 5:
                return f"**+{val:.1f}%** ✓"
            elif val < -5:
                return f"{val:.1f}% ✗"
            else:
                return f"{val:+.1f}%"

        md.append(f"| {policy_name:10s} | {format_pct(queue_imp)} | {format_pct(wait_imp)} | "
                  f"{format_pct(long_wait_imp)} | {format_pct(reward_imp)} |\n")

    md.append("\n---\n")

    # Statistical Significance
    md.append("## Statistical Significance\n")
    md.append(f"*T-test comparing episode rewards vs {baseline_name} (α=0.05)*\n\n")
    md.append("| Policy | Metric | p-value | Significant? |\n")
    md.append("|--------|--------|---------|-------------|\n")

    baseline_rewards = baseline["episode_rewards"]

    for policy_name, policy_data in results.items():
        if policy_name == baseline_name:
            continue

        policy_rewards = policy_data["episode_rewards"]
        test_result = statistical_test(baseline_rewards, policy_rewards)

        if "error" in test_result:
            md.append(f"| {policy_name:10s} | Reward | N/A | {test_result['error']} |\n")
        else:
            sig_str = "**YES**" if test_result["significant"] else "No"
            md.append(f"| {policy_name:10s} | Reward | {test_result['t_pval']:.4f} | {sig_str} |\n")

    md.append("\n---\n")

    # Action Patterns
    md.append("## Action Patterns\n")
    md.append("| Policy | Alternation Rate | Phase 0 Fraction |\n")
    md.append("|--------|------------------|------------------|\n")

    for policy_name, policy_data in results.items():
        s = policy_data["summary"]
        md.append(f"| {policy_name:10s} | {s['alternation_rate']:6.1f}% | {s['phase_0_fraction']:6.1f}% |\n")

    md.append("\n*Alternation Rate: % of steps where action changes (lower = more stable)*\n")
    md.append("\n---\n")

    # Interpretation
    md.append("## Interpretation\n\n")
    md.append("### Key Findings\n\n")

    # Find best policy by queue reduction
    best_policy = None
    best_queue_reduction = -float('inf')
    for policy_name, policy_data in results.items():
        if policy_name == baseline_name:
            continue
        s = policy_data["summary"]
        reduction = compute_improvement(baseline_summary["queue_mean"], s["queue_mean"], lower_is_better=True)
        if reduction > best_queue_reduction:
            best_queue_reduction = reduction
            best_policy = policy_name

    if best_policy:
        md.append(f"- **Best performing policy:** {best_policy} ({best_queue_reduction:+.1f}% queue reduction)\n")
    else:
        md.append(f"- No policy outperformed baseline\n")

    # Check if any policy is statistically significant
    any_significant = False
    for policy_name, policy_data in results.items():
        if policy_name == baseline_name:
            continue
        test_result = statistical_test(baseline_rewards, policy_data["episode_rewards"])
        if not "error" in test_result and test_result["significant"]:
            any_significant = True
            break

    if any_significant:
        md.append(f"- **Statistical significance:** Yes (p < 0.05 for at least one policy)\n")
    else:
        md.append(f"- **Statistical significance:** No significant differences detected\n")

    md.append("\n### Recommendations\n\n")

    if best_queue_reduction > 10:
        md.append(f"1. Deploy **{best_policy}** policy for congestion reduction\n")
        md.append(f"2. Monitor long-term performance and adjust parameters if needed\n")
    else:
        md.append(f"1. Further tuning needed - improvements are marginal\n")
        md.append(f"2. Consider ensemble or hybrid approaches\n")

    md.append("\n---\n")
    md.append(f"\n*Report generated by tools/generate_report.py*\n")

    # Write to file
    with open(output_path, 'w') as f:
        f.write(''.join(md))

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report from results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to comparison results JSON")
    parser.add_argument("--baseline", type=str, default="random",
                        help="Baseline policy name for comparisons")
    parser.add_argument("--out", type=str, default="comparison_report.md",
                        help="Output markdown file")
    args = parser.parse_args()

    print("="*80)
    print("COMPARISON REPORT GENERATOR")
    print("="*80)
    print(f"Results: {args.results}")
    print(f"Baseline: {args.baseline}")
    print(f"Output: {args.out}")

    # Load results
    print(f"\nLoading results...")
    results = load_results(args.results)
    print(f"✓ Loaded results for {len(results)} policies: {', '.join(results.keys())}")

    # Generate report
    print(f"\nGenerating markdown report...")
    success = generate_markdown_report(results, args.baseline, args.out)

    if success:
        print(f"✓ Report saved to: {args.out}")
        print(f"\nView report:")
        print(f"  cat {args.out}")
        print(f"  # or open in markdown viewer")
        print("="*80)
    else:
        print(f"✗ Report generation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
