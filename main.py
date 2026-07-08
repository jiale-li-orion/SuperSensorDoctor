"""SuperSenseDoctor — Agent Layer 启动入口"""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print(f"SuperSenseDoctor Agent Layer starting...")
    print(f"LLM: {config['llm']['provider']}/{config['llm']['model']}")


if __name__ == "__main__":
    main()
