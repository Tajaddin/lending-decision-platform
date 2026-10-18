"""Settings from environment variables (prefix LDP_). See .env.example for every variable."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LDP_", env_file=".env", extra="ignore")

    # Storage. Local folders by default; on Databricks these are Unity Catalog volume paths.
    raw_dir: Path = Path("data/raw")
    lake_root: str = "data/lake"
    training_dir: Path = Path("data/training")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")

    # Spark (ignored on Databricks, where the session already exists)
    spark_master: str = "local[*]"
    spark_driver_memory: str = "8g"
    spark_shuffle_partitions: int = 64

    # Out-of-time split: train on one year, choose the threshold and calibrate on the next, test on the last.
    train_year: int = 2023
    valid_year: int = 2024
    test_year: int = 2025

    # Serving
    model_name: str = "ldp"
    tfs_url: str = "http://localhost:8501"
    tfs_timeout_s: float = 2.0
    api_keys: SecretStr = Field(default=SecretStr(""), description="comma-separated client:key pairs")
    rate_limit_per_minute: int = 600
    max_batch: int = 64
    max_inflight: int = 48  # concurrent model calls per API worker before shedding load

    # Decision events (Pub/Sub) and their store (BigQuery). The Google clients pick up
    # PUBSUB_EMULATOR_HOST themselves; BigQuery needs an explicit endpoint for its emulator.
    gcp_project: str = "ldp-local"
    pubsub_topic: str = "decisions"
    pubsub_subscription: str = "decisions-bq-sink"
    pubsub_dead_letter_topic: str = "decisions-dead-letter"
    publish_events: bool = True
    bq_dataset: str = "lending"
    bq_table: str = "decisions"
    bq_endpoint: str | None = None

    # Monitoring
    pushgateway_url: str | None = None
    monitor_window_hours: int = 24
    monitor_min_rows: int = 500

    @field_validator("lake_root")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def lake(self, layer: str) -> str:
        """A Delta path, or `uc:catalog.schema.layer` when lake_root is `uc:catalog.schema` (Databricks)."""
        sep = "." if self.lake_root.startswith("uc:") else "/"
        return f"{self.lake_root}{sep}{layer}"

    def client_keys(self) -> dict[str, str]:
        """key -> client name. Malformed entries are rejected rather than silently ignored."""
        pairs: dict[str, str] = {}
        raw = self.api_keys.get_secret_value().strip()
        if not raw:
            return pairs
        for item in raw.split(","):
            client, sep, key = item.strip().partition(":")
            if not sep or not client or len(key) < 16:
                raise ValueError("LDP_API_KEYS entries must be client:key with keys of at least 16 characters")
            pairs[key] = client
        return pairs


@lru_cache
def get_settings() -> Settings:
    return Settings()
