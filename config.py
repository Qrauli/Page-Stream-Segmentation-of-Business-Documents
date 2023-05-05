from pydantic import BaseSettings


class Settings(BaseSettings):
    batch_size: int = 30


settings = Settings()