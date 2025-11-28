"""
Módulo de configuración centralizada
Carga y gestiona configuraciones desde archivos YAML
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Clase de configuración inmutable"""

    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Permite acceso tipo diccionario"""
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene valor con default opcional"""
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Permite uso de 'in' operator"""
        return key in self.data


def load_config(config_path: str) -> Config:
    """
    Carga configuración desde archivo YAML

    Args:
        config_path: Ruta al archivo YAML de configuración

    Returns:
        Config: Objeto de configuración

    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si el archivo no es YAML válido
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    return Config(data=config_data)


def merge_configs(*configs: Config) -> Config:
    """
    Combina múltiples configuraciones
    Las configuraciones posteriores sobrescriben las anteriores

    Args:
        *configs: Configuraciones a combinar

    Returns:
        Config: Configuración combinada
    """
    merged_data = {}

    for config in configs:
        merged_data.update(config.data)

    return Config(data=merged_data)


def save_config(config: Config, output_path: str) -> None:
    """
    Guarda configuración en archivo YAML

    Args:
        config: Configuración a guardar
        output_path: Ruta del archivo de salida
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config.data, f, default_flow_style=False, allow_unicode=True)
