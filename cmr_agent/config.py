from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    anthropic_api_key: str | None = Field(default=None, alias='ANTHROPIC_API_KEY')
    cmr_base_url: str = Field(default='https://cmr.earthdata.nasa.gov', alias='CMR_BASE_URL')
    cmr_provider: str = Field(default='ALL', alias='CMR_PROVIDER')
    vector_db_dir: str = Field(default='./vectordb/chroma', alias='VECTOR_DB_DIR')

    # Normalize provider so defaults behave consistently even if a local .env sets legacy values
    @field_validator('cmr_provider', mode='before')
    @classmethod
    def normalize_provider(cls, value: str | None):
        if value is None:
            return 'ALL'
        v = str(value).strip().upper()
        if v in ('', 'CMR', 'CMR_ALL'):
            return 'ALL'
        return v

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()

