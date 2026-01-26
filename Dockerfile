# Use the official uv image for Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation and UTF-8 locale
ENV UV_COMPILE_BYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Copy requirements file first for better caching
COPY requirements.txt .

# Install dependencies
RUN uv pip install --no-cache -r requirements.txt --system

# Copy the rest of the application
COPY . .

# Set the command to run the ETL script
CMD ["uv", "run", "python", "ETL_expr/news_extract.py"]
