try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DataCleaningAction, DataCleaningObservation
    from .data_cleaning_env_environment import DataCleaningEnvironment
except ImportError:
    try:
        from models import DataCleaningAction, DataCleaningObservation
        from server.data_cleaning_env_environment import DataCleaningEnvironment
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from models import DataCleaningAction, DataCleaningObservation
        from data_cleaning_env_environment import DataCleaningEnvironment


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data_cleaning_env",
    max_concurrent_envs=1,
)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main()