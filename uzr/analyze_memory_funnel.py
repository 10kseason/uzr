# analyze_memory_funnel.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def analyze_decision_funnel(csv_path="memory_decisions.csv"):
    """Analyze memory decision funnel and conversion rates.

    Args:
        csv_path: Path to memory decisions CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping decision funnel analysis.")
        return None

    df = pd.read_csv(csv_path)

    counts = df['decision'].value_counts()
    total = len(df)

    print("=== Memory Decision Funnel ===")
    for decision, count in counts.items():
        pct = count / total * 100
        print(f"{decision:12s}: {count:6d} ({pct:5.2f}%)")

    # Stage → Promote conversion rate
    stage_count = counts.get('stage', 0)

    if os.path.exists("logus-luria/shadow_bank_events.jsonl"):
        # Count promote events from shadow bank events
        import json
        promote_count = 0
        with open("logus-luria/shadow_bank_events.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get("action") == "promote":
                        promote_count += event.get("count", 1)
                except:
                    pass

        if stage_count > 0:
            conversion = promote_count / stage_count * 100
            loss = 100 - conversion
            print(f"\n=== Stage Conversion ===")
            print(f"Total staged: {stage_count}")
            print(f"Promoted: {promote_count} ({conversion:.1f}%)")
            print(f"Lost to decay: {stage_count - promote_count} ({loss:.1f}%)")

    return df


def plot_shadow_dynamics(csv_path="logus-luria/shadow_bank_stats.csv"):
    """Visualize shadow bank dynamics over time.

    Args:
        csv_path: Path to shadow bank statistics CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping shadow dynamics plot.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Size over time
    axes[0, 0].plot(df['step'], df['shadow_size'])
    axes[0, 0].set_title('Shadow Bank Size')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Size')

    # Promote ratio
    if 'promote_ratio' in df.columns:
        axes[0, 1].plot(df['step'], df['promote_ratio'])
        axes[0, 1].set_title('Promote Ratio')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Ratio')

    # Average surprise
    if 'shadow_surprise_mean' in df.columns:
        axes[1, 0].plot(df['step'], df['shadow_surprise_mean'])
        axes[1, 0].set_title('Average Shadow Surprise')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Surprise')

    # Average age
    if 'shadow_mean_age' in df.columns:
        axes[1, 1].plot(df['step'], df['shadow_mean_age'])
        axes[1, 1].set_title('Average Item Age')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Age (steps)')

    plt.tight_layout()
    plt.savefig('shadow_dynamics.png')
    print("Saved: shadow_dynamics.png")


def plot_dedup_stats(csv_path="logus-luria/dedup_stats.csv"):
    """Visualize deduplication statistics over time.

    Args:
        csv_path: Path to deduplication statistics CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping dedup stats plot.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Duplicate rate over time
    axes[0].plot(df['step'], df['dup_rate'])
    axes[0].set_title('Duplicate Rate Over Time')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Dup Rate')

    # Average similarity
    if 'avg_sim_candidates' in df.columns:
        axes[1].plot(df['step'], df['avg_sim_candidates'], label='Avg Sim')
        if 'p95_sim_candidates' in df.columns:
            axes[1].plot(df['step'], df['p95_sim_candidates'], label='P95 Sim')
        axes[1].set_title('Candidate Similarity')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Similarity')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig('dedup_stats.png')
    print("Saved: dedup_stats.png")


def analyze_stage_transitions(jsonl_path="logus-luria/stage_events.jsonl"):
    """Analyze stage transitions from event log.

    Args:
        jsonl_path: Path to stage events JSONL file
    """
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found. Skipping stage transition analysis.")
        return

    import json

    print("\n=== Stage Transitions ===")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
                print(f"Step {event['step']:6d}: {event['stage']:20s} ({event['reason']})")
            except:
                pass


if __name__ == "__main__":
    print("=" * 60)
    print("UZR Memory System Analysis")
    print("=" * 60)

    # Decision funnel analysis
    df = analyze_decision_funnel()

    # Shadow bank dynamics visualization
    plot_shadow_dynamics()

    # Deduplication statistics
    plot_dedup_stats()

    # Stage transitions
    analyze_stage_transitions()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
