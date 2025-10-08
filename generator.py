import logging
from src.generator.core.generator import ConfigurationGenerator
from src.generator.core.config import GeneratorConfig


def main():
    """Main entry point for the configuration generator."""
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(console_handler)

    # Set DEBUG level only for our own packages
    logging.getLogger('src').setLevel(logging.DEBUG)

    # Create configuration generator
    config = GeneratorConfig(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0
    )
    generator = ConfigurationGenerator(config)

    # Example repository URLs to process
    repo_url = "https://github.com/run-rasztabiga-me/poc1-fastapi.git"
    # repo_url = "https://github.com/run-rasztabiga-me/poc2-fastapi.git"

    # Generate configurations
    config_output, messages = generator.generate(repo_url)
    print(f"Generated files:")
    print(f"  Dockerfiles: {config_output.dockerfiles}")
    print(f"  Kubernetes files: {config_output.kubernetes_files}")


if __name__ == "__main__":
    main()
