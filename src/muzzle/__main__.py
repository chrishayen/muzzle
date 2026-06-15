from __future__ import annotations

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "muzzle.app:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()

