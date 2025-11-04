"""
Main entry point for the MyAgent Continuous AI App Builder.
Run with: python -m core.orchestrator.continuous_director
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from config.logging_config import setup_logging
from core.orchestrator.continuous_director import ContinuousDirector


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MyAgent - Continuous AI App Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with a simple project
  python -m core --project my-app --spec '{\"name\": \"My App\"}'

  # Start with debug logging
  python -m core --project my-app --debug

  # Resume from checkpoint
  python -m core --project my-app --resume

  # Use specific LLM provider
  python -m core --project my-app --provider anthropic
        """
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name"
    )

    parser.add_argument(
        "--spec",
        type=str,
        default="{}",
        help="Project specification as JSON string"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider to use (overrides .env setting)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (for testing)"
    )

    return parser.parse_args()


async def main():
    """Main application entry point"""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        log_level = "DEBUG" if args.debug else "INFO"
        setup_logging(level=log_level)

        logger.info("=" * 80)
        logger.info(f"MyAgent Continuous AI App Builder v{settings.VERSION}")
        logger.info("=" * 80)

        # Override LLM provider if specified
        if args.provider:
            settings.DEFAULT_LLM_PROVIDER = args.provider
            logger.info(f"Using LLM provider: {args.provider}")

        # Validate configuration
        logger.info("Validating configuration...")
        try:
            if settings.DEFAULT_LLM_PROVIDER in ["openai", "anthropic"]:
                settings.validate_required_keys()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.error("Please set the appropriate API key in your .env file")
            sys.exit(1)

        # Ensure directories exist
        settings.ensure_directories()
        logger.success("Configuration validated")

        # Parse project spec
        import json
        try:
            project_spec = json.loads(args.spec)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in project spec: {args.spec}")
            sys.exit(1)

        # Create the continuous director
        logger.info(f"Initializing project: {args.project}")
        director = ContinuousDirector(
            project_name=args.project,
            project_spec=project_spec
        )

        # Start the continuous development process
        logger.info("Starting continuous development...")
        logger.info("Press Ctrl+C to stop gracefully")

        if args.max_iterations:
            logger.warning(f"Maximum iterations set to {args.max_iterations}")

        await director.start()

        logger.success("Continuous development completed!")

    except KeyboardInterrupt:
        logger.warning("\nReceived interrupt signal. Stopping gracefully...")
        if 'director' in locals():
            await director.stop()
        logger.info("Stopped successfully")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
