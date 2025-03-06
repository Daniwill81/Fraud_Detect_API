import base64
import binascii
import typing

import jwt
from beanie import operators
from fastapi import Request, status
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED as HTTP_401

from sap.beanie import Document
from sap.beanie.exceptions import Object404Error
from sap.fastapi.auth import BasicAuth

from app.models.enums import RoleEnum
from AppMain.settings import logger

from .user import User


class UserAuth(BasicAuth):
    """
    BasicAuth for API request as user.

    This is used by the React Frontend.
    """

    auth_key_attribute: typing.ClassVar[str] = "auth_key"

    async def authenticate(self, request: Request) -> User:
        """Provide the authenticated user to views that require it."""
        assert self.user_model and issubclass(self.user_model, Document)

        header_auth: str | None = request.headers.get("Authorization") or request.headers.get("X-Beans-Authorization")
        logger.debug("DEBUGGING HEADERS => %s", str(request.headers))

        if not header_auth:
            raise HTTPException(HTTP_401, detail="Authentication required")

        scheme, credentials = header_auth.split()
        if scheme.lower() != "basic":
            raise HTTPException(HTTP_401, detail="Only basic authorization is supported")

        try:
            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
            raise HTTPException(HTTP_401, detail="Error while decoding basic auth credentials") from exc

        username, _, pwd = decoded.partition(":")
        user_key = username or pwd

        if auth_key_name := self.get_auth_key_attribute():
            auth_key = getattr(self.user_model, auth_key_name)
        else:
            auth_key = None

        try:
            if auth_key:
                return await self.user_model.find_one_or_404(
                    operators.Or(operators.Eq(User.auth_key, user_key), operators.Eq(User.auth_key, user_key))
                )
            return await self.user_model.get_or_404(user_key)
        except (Object404Error, jwt.exceptions.InvalidTokenError) as exc:
            raise HTTPException(HTTP_401, detail="Invalid basic auth credentials") from exc

    def require(
        self, perms: typing.Union[list[str], list[RoleEnum]]
    ) -> typing.Callable[[Request], typing.Awaitable[User]]:
        """Authenticate the user and check that they have enough privilege to access resource."""

        async def auth_with_perms(request: Request) -> User:
            user: User = await self.authenticate(request=request)
            if user.role not in perms:
                # if not user.has_perms(perms):
                raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not enough permission to access resource")

            return user

        return auth_with_perms


user_auth = UserAuth(user_model=User)
