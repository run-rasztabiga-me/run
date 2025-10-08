import logging
import sys
from typing import List

from src.evaluator.core.evaluator import ConfigurationEvaluator
from src.generator.core.config import GeneratorConfig


def setup_logging():
    """Configure logging for the evaluator."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Enable detailed logging from our code
    logging.getLogger('src').setLevel(logging.INFO)

    # Set log level for specific loggers to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('docker').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)


def main():
    """Main entry point for the evaluator."""
    setup_logging()
    logger = logging.getLogger(__name__)

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

    logger.info("🚀 Starting Configuration Generation Evaluation")
    logger.info(f"📋 Evaluating {len(test_repos)} repositories")
    logger.info("=" * 60)

    try:
        # Evaluate each repository individually and save detailed reports
        for i, repo_url in enumerate(test_repos, 1):
            logger.info(f"\n[{i}/{len(test_repos)}] Processing: {repo_url}")
            try:
                evaluate_single_repo(repo_url)
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {repo_url}: {str(e)}", exc_info=True)
                # Continue with next repository

        logger.info("\n" + "=" * 60)
        logger.info("✅ Evaluation completed!")
        logger.info(f"📁 Individual reports saved to: ./evaluation_reports/")

    except KeyboardInterrupt:
        logger.warning("⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


def evaluate_single_repo(repo_url: str):
    """Evaluate a single repository and save detailed report."""
    # Only setup logging if not already configured (when called directly)
    if not logging.getLogger().handlers:
        setup_logging()

    logger = logging.getLogger(__name__)

    generator_config = GeneratorConfig(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0
    )

    evaluator = ConfigurationEvaluator(
        generator_config=generator_config
    )

    logger.info(f"🔍 Evaluating: {repo_url}")
    report = evaluator.evaluate_repository(repo_url)

    # Save individual report
    report_file = evaluator.save_report(report)
    logger.info(f"📄 Report saved to: {report_file}")

    return report


if __name__ == "__main__":
    # Check if a specific repo URL was provided as argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        repo_url = sys.argv[1]
        evaluate_single_repo(repo_url)
    else:
        main()
