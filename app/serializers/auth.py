"""
# Serializers.

Handle data validation.
"""

import re
import typing

import pydantic
from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import Field

from sap.fastapi import WriteObjectSerializer
from sap.fastapi.auth import JWTAuth

from app.models import User
from app.serializers.user import UserSerializer

jwt_auth = JWTAuth(user_model=User)


class AuthTokenSerializer(pydantic.BaseModel):
    """Serializer the authentication token."""

    id: str
    auth_key: str
    user: UserSerializer

    @classmethod
    async def get(cls, pk: str) -> User | None:
        """Used to verify that the token has been reset in the db"""
        return await User.find_one(User.auth_key == pk)


class LoginAuthSerializer(WriteObjectSerializer[User]):
    """
    Login form.

    Allow the user who have authenticate to the web app using email and password.
    """

    email: pydantic.EmailStr
    password: str

    instance: User | None = None

    async def run_async_validators(self, **kwargs: typing.Any) -> None:
        """Check that data pass DB validation."""
        await self.verify_email_password()

    async def verify_email_password(self) -> bool:
        """Check that the email exists in the DB."""
        self.instance = await User.find_current(email=self.email)

        if self.instance is not None and self.instance.verify_password(self.password):
            return True

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Les identifiants que vous avez saisi ne sont pas valides. "
                "Si vous avez oublié votre mot de passe, veuillez procéder à sa réinitialisation."
            ),
        )


class ForgotPasswordSerializer(WriteObjectSerializer[User]):
    """
    Password forgotten form.

    Allow the user who have forgotten their password to request a reset link.
    """

    email: pydantic.EmailStr

    instance: User | None = None

    async def run_async_validators(self, **kwargs: typing.Any) -> None:
        """Check that data pass DB validation."""
        await self.verify_email_exists()

    async def verify_email_exists(self) -> bool:
        """Check that the email exists in the DB."""
        self.instance: User = await User.find_current(email=self.email)
        return True


class ResetPasswordSerializer(WriteObjectSerializer[User]):
    """
    Password reset validation.

    Allow the user to set a new password.
    """

    password: str = Field(min_length=8)
    password2: str = Field(min_length=8)

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

    @pydantic.model_validator(mode="after")
    def validate_passwords_equals(self) -> typing.Self:
        """Verify that both password are equals."""
        assert self.password == self.password2, "Le mot de passe de confirmation ne corresponds pas."
        return self

    async def update(self, **kwargs: typing.Any) -> User:
        """Reset the password and verify the user."""
        assert self.instance
        self.instance.set_password(self.password)
        await self.instance.save()

        return self.instance
