"""
# Serializers.

Handle data validation.
"""

import datetime
import re
import typing

import pydantic
from pydantic import Field

from sap.fastapi import ObjectSerializer, WriteObjectSerializer

from app.models.enums import RoleEnum, SexEnum
from app.models.user.user import User


class UserSerializer(ObjectSerializer[User]):
    """Serialize the `user` object for retrieve and listing."""

    id: str
    first_name: str
    last_name: str
    email: pydantic.EmailStr
    birthdate: datetime.date | None = None
    sex: SexEnum | None = None
    role: RoleEnum
    created: datetime.datetime


class WriteUserSerializer(WriteObjectSerializer[User]):
    """Serialize the `user` object for create and update."""

    first_name: str
    last_name: str
    email: pydantic.EmailStr
    birthdate: datetime.date | None = None
    sex: SexEnum | None = None
    password: str = Field(min_length=8)
    role: RoleEnum

    # The fields bellow are not serialized
    instance: User | None = None

    @pydantic.field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """Verify that the password fit security criteria"""
        if not re.findall(r"[a-zA-Z]+", value):
            raise AssertionError("Votre mot de passe doit contenir au moins une lettre.")
        if not re.findall(r"[0-9]+", value):
            raise AssertionError("Votre mot de passe doit contenir au moins 1 chiffre.")
        if re.match(r"^\w+$", value):
            raise AssertionError("Votre mot de passe doit contenir au moins 1 caractère spécial.")
        return value

    async def run_async_validators(self, **kwargs: typing.Any) -> None:
        """Check that data pass DB validation."""
        await self.validate_email_uniqueness()
        await self.validate_role()

    async def validate_email_uniqueness(self) -> None:
        """Check that the email is not used by another user."""
        if self.instance and self.instance.email == self.email:
            return

        is_duplicate = await User.find_current(email=self.email)
        if is_duplicate:
            raise AssertionError("Cet email est déjà rattaché à un compte administrateur existant.")

    async def validate_role(self) -> None:
        """Set user role."""
        if self.instance and self.role == self.instance.role:
            return

    async def create(self, **kwargs: typing.Any) -> User:
        """Create users."""
        user = User(
            first_name=self.first_name,
            last_name=self.last_name,
            birthdate=self.birthdate,
            sex=self.sex,
            email=self.email,
            role=self.role,
        )
        user.set_password(self.password)
        await user.create()
        self.instance = user
        return user

    async def update(self, **kwargs: typing.Any) -> User:
        """Update the object in the database using the data extracted by the serializer."""
        assert self.instance
        data_to_update = {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "birthdate": self.birthdate,
            "sex": self.sex,
            "email": self.email,
        }
        instance: User = self.instance.model_copy(update=data_to_update)
        await instance.save()
        self.instance = instance
        return instance
