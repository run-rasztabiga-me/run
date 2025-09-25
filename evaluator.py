import logging
import sys
from typing import List

from src.evaluator.core.evaluator import ConfigurationEvaluator
from src.generator.core.config import GeneratorConfig


def main():
    """Main entry point for the evaluator."""
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Set log level for specific loggers to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('docker').setLevel(logging.WARNING)

    # Repository URLs to evaluate (matching thesis test repositories)
    test_repos = [
        "https://github.com/run-rasztabiga-me/poc1-fastapi.git",
        # "https://github.com/run-rasztabiga-me/poc2-fastapi.git",
    ]

    # Additional test repositories (commented out for now)
    # test_repos = test_repos + [
    #     "https://github.com/enriquecatala/fastapi-helloworld.git",
    #     "https://github.com/Azure-Samples/azd-simple-fastapi-appservice.git",
    #     "https://github.com/carvalhochris/fastapi-htmx-hello.git",
    #     "https://github.com/Sivasuthan9/fastapi-docker-optimized.git",
    #     "https://github.com/renceInbox/fastapi-todo.git",
    #     "https://github.com/beerjoa/fastapi-postgresql-boilerplate.git",
    # ]

    # Create evaluator with configuration
    generator_config = GeneratorConfig(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0
    )

    evaluator = ConfigurationEvaluator(
        generator_config=generator_config
    )

    print(f"🚀 Starting Configuration Generation Evaluation")
    print(f"📋 Standard Evaluation: {len(test_repos)} repositories")
    print("=" * 60)

    try:
        # Standard batch evaluation
        reports = evaluator.evaluate_batch(test_repos)

        # Export results
        print("\n📊 Generating evaluation reports...")
        exported_files = evaluator.export_batch_results(reports)

        print("\n✅ Evaluation completed!")
        print(f"📁 Reports exported to:")
        for format_type, filepath in exported_files.items():
            print(f"  - {format_type.upper()}: {filepath}")

        # Print summary to console
        print("\n📈 Summary:")
        successful = sum(1 for r in reports if r.generation_result and r.generation_result.success)
        print(f"  - Total repositories: {len(reports)}")
        print(f"  - Successful generations: {successful}")
        print(f"  - Success rate: {(successful/len(reports)*100):.1f}%")

        if successful > 0:
            avg_time = sum(
                r.generation_result.generation_time for r in reports
                if r.generation_result and r.generation_result.generation_time
            ) / successful
            avg_score = sum(
                r.quality_metrics.overall_score for r in reports
                if r.quality_metrics and r.quality_metrics.overall_score
            ) / successful if any(r.quality_metrics for r in reports) else 0
            print(f"  - Average generation time: {avg_time:.1f}s")
            print(f"  - Average quality score: {avg_score:.1f}/100")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


def evaluate_single_repo(repo_url: str):
    """Evaluate a single repository (utility function)."""
    generator_config = GeneratorConfig(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0
    )

    evaluator = ConfigurationEvaluator(
        generator_config=generator_config
    )

    print(f"🔍 Evaluating: {repo_url}")
    report = evaluator.evaluate_repository(repo_url)

    # Save individual report
    report_file = evaluator.save_report(report)
    print(f"📄 Report saved to: {report_file}")

    return report


if __name__ == "__main__":
    # Check if a specific repo URL was provided as argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        repo_url = sys.argv[1]
        evaluate_single_repo(repo_url)
    else:
        main()
