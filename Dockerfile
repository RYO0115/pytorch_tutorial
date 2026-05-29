FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl

# uvのインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# プロジェクトファイルのコピーと依存関係インストール
COPY pyproject.toml .
RUN uv sync --system
