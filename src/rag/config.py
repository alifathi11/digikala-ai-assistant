from pathlib import Path

import yaml


PROJECT_CONFIG_NAME = "project.yaml"


def load_config(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_project_config(project_root):
    return load_config(
        Path(project_root)
        / "configs"
        / PROJECT_CONFIG_NAME
    )


def get_model_pricing(config, model):
    pricing = (
        config.get(
            "models",
            {},
        ).get(
            str(model),
        )
    )

    if pricing is None:
        raise KeyError(
            f"Missing token pricing for model: {model}"
        )

    return {
        "input_cost_per_million": float(
            pricing[
                "input_cost_per_million"
            ]
        ),
        "output_cost_per_million": float(
            pricing[
                "output_cost_per_million"
            ]
        ),
    }
