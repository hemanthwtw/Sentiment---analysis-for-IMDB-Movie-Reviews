import os

# Keep secrets configurable via environment variables for production safety.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "336f10a58358032815003270c8c78f78")
TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN", "")
