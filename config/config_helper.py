import pathlib
import yaml
import json
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def get_model_config():
    path = pathlib.Path(__file__).parent.joinpath("config.yaml")
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

            choice = config.get("model_choice")
            if not choice:
                raise ValueError("Config 'model_choice' not set in config.yaml")

            all_models = config.get("models")
            if not all_models:
                raise ValueError("Config 'models' section not set in config.yaml")

            provider_choice = all_models.get(choice)
            if not provider_choice:
                raise ValueError(f"Model '{choice}' not found in 'models' section.")

            return provider_choice

    except FileNotFoundError:
        logging.error("Storage File Not Found")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Yaml Error: {e}")
        return None
    except KeyError as e:
        logging.error(f"Key Error: {e}")
        return None
    except TypeError as e:
        logging.error(f"Type Error {e}")
        return None
    except Exception as e:
        logging.error(f"Unknown Error: {e}")
        return None