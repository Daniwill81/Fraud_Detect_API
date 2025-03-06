# pylint: disable=import-outside-toplevel, broad-exception-caught

"""
Migrate.

Run database migrations.
"""
import asyncio
import typing

from beanie import operators
from motor.motor_asyncio import AsyncIOMotorDatabase

from sap.beanie.client import BeanieClient

from app import controllers
from AppMain.asgi import initialize_beanie


async def run_migrations() -> None:
    """Run data migrations."""
    await initialize_beanie()


if __name__ == "__main__":
    asyncio.run(run_migrations())
