from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "anthropic"
    llm_model_name: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.3
    max_reflection_cycles: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
