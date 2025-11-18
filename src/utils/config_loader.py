"""
Configuration loader for YAML files
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage YAML configuration files"""

    def __init__(self, config_path: Path):
        """
        Initialize config loader

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[OK] Loaded configuration from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.hidden_dim')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value

        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.debug(f"Updated config: {key} = {value}")

    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save configuration to YAML file

        Args:
            output_path: Output path. If None, overwrites original file
        """
        if output_path is None:
            output_path = self.config_path

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"[OK] Saved configuration to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path={self.config_path})"

    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Convenience function to load YAML config

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.to_dict()


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
