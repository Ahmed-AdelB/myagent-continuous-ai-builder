"""
Settings and configuration management for MyAgent.
Loads environment variables and provides validated configuration.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Project metadata
    PROJECT_NAME: str = "MyAgent"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://myagent:myagent_password@localhost:5432/myagent_db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")

    # ChromaDB Configuration
    CHROMADB_HOST: str = Field(default="localhost", env="CHROMADB_HOST")
    CHROMADB_PORT: int = Field(default=8000, env="CHROMADB_PORT")
    CHROMADB_PATH: str = Field(
        default="persistence/vector_memory",
        env="CHROMADB_PATH"
    )

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL"
    )

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_MODEL: str = Field(default="gpt-5-chat-latest", env="DEFAULT_MODEL")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=2000, env="LLM_MAX_TOKENS")

    # Quality Targets
    TARGET_TEST_COVERAGE: float = Field(default=95.0, env="TARGET_TEST_COVERAGE")
    TARGET_PERFORMANCE_SCORE: float = Field(
        default=90.0,
        env="TARGET_PERFORMANCE_SCORE"
    )
    TARGET_DOCUMENTATION_COVERAGE: float = Field(
        default=90.0,
        env="TARGET_DOCUMENTATION_COVERAGE"
    )
    TARGET_CODE_QUALITY_SCORE: float = Field(
        default=85.0,
        env="TARGET_CODE_QUALITY_SCORE"
    )
    TARGET_SECURITY_SCORE: float = Field(default=95.0, env="TARGET_SECURITY_SCORE")
    TARGET_USER_SATISFACTION: float = Field(
        default=90.0,
        env="TARGET_USER_SATISFACTION"
    )

    # Agent Configuration
    MAX_AGENT_RETRIES: int = Field(default=3, env="MAX_AGENT_RETRIES")
    AGENT_TIMEOUT_SECONDS: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    CHECKPOINT_INTERVAL_HOURS: int = Field(
        default=1,
        env="CHECKPOINT_INTERVAL_HOURS"
    )

    # Persistence
    PERSISTENCE_DIR: Path = Field(
        default=Path("persistence"),
        env="PERSISTENCE_DIR"
    )
    LOGS_DIR: Path = Field(default=Path("logs"), env="LOGS_DIR")

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(
        default=60,
        env="JWT_EXPIRATION_MINUTES"
    )

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("PERSISTENCE_DIR", "LOGS_DIR", pre=True)
    def parse_path(cls, v):
        """Convert string paths to Path objects"""
        if isinstance(v, str):
            return Path(v)
        return v

    def validate_required_keys(self) -> bool:
        """Validate that required API keys are present"""
        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        if self.DEFAULT_LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using Anthropic provider"
            )
        return True

    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.PERSISTENCE_DIR,
            self.PERSISTENCE_DIR / "database",
            self.PERSISTENCE_DIR / "vector_memory",
            self.PERSISTENCE_DIR / "checkpoints",
            self.PERSISTENCE_DIR / "agents",
            self.PERSISTENCE_DIR / "snapshots",
            self.LOGS_DIR,
            self.LOGS_DIR / "agents",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


# Validate settings on import if not in test mode
if not os.getenv("TESTING"):
    try:
        if settings.DEFAULT_LLM_PROVIDER in ["openai", "anthropic"]:
            settings.validate_required_keys()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Set the appropriate API key in your .env file or use Ollama")
