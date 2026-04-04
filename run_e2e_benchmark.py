#!/usr/bin/env python3
"""
Quick E2E benchmark runner with focused comparison (Interactive Version).

Runs a small-scale evaluation comparing:
1. Full pipeline (all features)
2. No decomposition (single query)
3. Database-only (no external APIs)
4. Baseline (minimal features)

Database Isolation:
    Option 1 (Recommended): Separate test database
        1. docker-compose -f docker-compose.dev.yml up -d postgres-test
        2. Set: export BEIR_TEST_DATABASE_URL="postgresql+asyncpg://exegent_test:test_password_123@localhost:5433/beir_test"
        3. python run_e2e_benchmark.py
    
    Option 2: Separate table in main database
        - Just run: python run_e2e_benchmark.py
        - Uses beir_test_papers table
        - Production papers table NOT affected
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.evaluation.e2e_pipeline_benchmark import (
    ExegentPipelineBenchmark,
    PipelineVariant,
)
from app.db.database import db_session_context


class BenchmarkConfig:
    """Simple class to hold our configuration, acting like argparse.Namespace"""
    def __init__(self):
        self.dataset = "scifact"
        self.max_queries = 300
        self.variants = None
        self.decomposition = "off"
        self.compare_rerank = False
        self.test_db_url = None
        self.yes = False


def get_interactive_inputs() -> BenchmarkConfig:
    """Prompt the user for configuration step-by-step."""
    args = BenchmarkConfig()
    
    print("=" * 80)
    print("Interactive Exegent E2E Pipeline Benchmark Setup")
    print("Press Enter to accept the [default] values.")
    print("=" * 80 + "\n")

    # 1. Dataset
    dataset_input = input("1. Choose dataset (scifact, nfcorpus, scidocs) [scifact]: ").strip().lower()
    if dataset_input:
        while dataset_input not in ["scifact", "nfcorpus", "scidocs"]:
            print("   Invalid choice.")
            dataset_input = input("   Choose dataset (scifact, nfcorpus, scidocs) [scifact]: ").strip().lower()
        args.dataset = dataset_input

    # 2. Max Queries
    max_queries_input = input("2. Max queries to test [300 (Full scifact)]: ").strip()
    if max_queries_input:
        while True:
            try:
                args.max_queries = int(max_queries_input)
                break
            except ValueError:
                print("   Please enter a valid integer.")
                max_queries_input = input("   Max queries to test [50]: ").strip()

    # 3. Decomposition
    decomp_input = input("3. Decomposition mode (both, on, off) [off]: ").strip().lower()
    if decomp_input:
        while decomp_input not in ["both", "on", "off"]:
            print("   Invalid choice.")
            decomp_input = input("   Decomposition mode (both, on, off) [off]: ").strip().lower()
        args.decomposition = decomp_input

    # 4. Compare Rerank
    rerank_input = input("4. Compare before/after rerank? (y/n) [n]: ").strip().lower()
    if rerank_input in ['y', 'yes']:
        args.compare_rerank = True

    return args


def resolve_variants(args: BenchmarkConfig) -> list[PipelineVariant]:
    """Resolve variants exactly like the benchmark module default logic."""
    if args.variants:
        return [PipelineVariant(v) for v in args.variants] # type: ignore

    base_variants: list[PipelineVariant] = []
    if args.decomposition in ("both", "on"):
        base_variants.extend(
            [
                PipelineVariant.DECOMPOSED_BEFORE_RERANK,
                PipelineVariant.DECOMPOSED_AFTER_RERANK,
            ]
        )
    if args.decomposition in ("both", "off"):
        base_variants.extend(
            [
                PipelineVariant.ORIGINAL_BEFORE_RERANK,
                PipelineVariant.ORIGINAL_AFTER_RERANK,
            ]
        )

    if not args.compare_rerank:
        base_variants = [v for v in base_variants if "after_rerank" in v.value]

    return base_variants


async def main():
    # Gather inputs interactively instead of parsing command-line args
    args = get_interactive_inputs()
    variants = resolve_variants(args)

    print("\n" + "=" * 80)
    print("Benchmark configuration summary:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Max queries: {args.max_queries}")
    print(f"- Variants: {[v.value for v in variants]}")
    
    # Check database configuration
    test_db_url = args.test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")
    if test_db_url:
        print("\n✅ Using SEPARATE TEST DATABASE (complete isolation)")
        print("   Production database will NOT be touched.")
    else:
        print("\n⚠️  Using SEPARATE TABLE in main database (beir_test_papers)")
        print("   Production 'papers' table will NOT be affected.")
        print("\n   For complete isolation, use separate database:")
        print("   1. docker-compose -f docker-compose.dev.yml up -d postgres-test")
        print("   2. export BEIR_TEST_DATABASE_URL=\"postgresql+asyncpg://exegent_test:test_password_123@localhost:5433/beir_test\"")
        print("   3. Re-run this script")
    
    print("\n" + "=" * 80)
    response = input("Continue with this configuration? [Y/n]: ").strip().lower()
    if response in ['n', 'no']:
        print("Aborted.")
        return
    
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")
    
    try:
        # Avoid overriding the explicitly provided URL if we pulled from env
        final_db_url = args.test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")
        
        if final_db_url:
            benchmark = ExegentPipelineBenchmark(
                dataset_name=args.dataset,
                test_db_url=final_db_url,
            )

            results = await benchmark.run_full_evaluation(
                variants=variants,
                max_queries=args.max_queries,
            )
        else:
            async with db_session_context() as session:
                benchmark = ExegentPipelineBenchmark(
                    db_session=session,
                    dataset_name=args.dataset,
                )

                results = await benchmark.run_full_evaluation(
                    variants=variants,
                    max_queries=args.max_queries,
                )
            
            print("\n" + "=" * 80)
            print("✓ Evaluation Complete!")
            print("=" * 80)

            print("\n📊 Key Insights:")
            for name, result in results.items():
                print(
                    f"  {name}: NDCG@10={result.avg_metrics.ndcg_at_10:.4f}, "
                    f"Recall@20={result.avg_metrics.recall_at_20:.4f}, "
                    f"MRR={result.avg_metrics.mrr:.4f}, "
                    f"Latency={result.avg_metrics.latency_ms:.0f}ms"
                )
            
            print("\n📁 Results saved to evaluation_results/e2e/\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure database is running")
        print("  2. Check .env configuration")
        print("  3. Verify LLM service is accessible")
        print("  4. Check logs for detailed errors")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())