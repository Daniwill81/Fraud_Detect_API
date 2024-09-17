# pylint: disable=import-outside-toplevel, broad-exception-caught

"""
Migrate.

Run database migrations.
"""
import asyncio

from AppMain.asgi import initialize_beanie


async def run_migrations() -> None:
    """Run data migrations."""
    await initialize_beanie()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_migrations())
