"""
# WebAPI.

The API endpoint  is queried by other applications
to communicate with this application. This endpoint usually relies
on a header based authenticated encoded in the request headers.
Commonly Basic or Bearer Auth.

It should accept and returns data formatted in JSON.

The API is structured with  Representational state transfer architecture:
https://en.wikipedia.org/wiki/Representational_state_transfer
"""

from fastapi import APIRouter, Request, status

from app.models import User
from app.serializers.auth import (
    AuthTokenSerializer,
    ForgotPasswordSerializer,
    LoginAuthSerializer,
    ResetPasswordSerializer,
)
from app.serializers.user import UserSerializer

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create(request: Request, serializer_write: LoginAuthSerializer) -> AuthTokenSerializer:
    """Create an authentication token."""
    await serializer_write.run_async_validators(request=request)
    assert (user := serializer_write.instance)
    await user.generate_auth_key()
    assert user.auth_key
    return AuthTokenSerializer(id=user.auth_key, auth_key=user.auth_key, user=UserSerializer.read(user))


@router.post("/reset_password/", status_code=status.HTTP_202_ACCEPTED)
async def reset_password(request: Request, serializer_write: ResetPasswordSerializer) -> AuthTokenSerializer:
    """Let users to set a new password.

    This is also used to confirm registration of new users.
    Since registration is confirmed by updating the password.
    """
    await serializer_write.run_async_validators(request=request)
    user = await serializer_write.update()
    await user.generate_auth_key()
    assert user.auth_key
    return AuthTokenSerializer(id=user.auth_key, auth_key=user.auth_key, user=UserSerializer.read(user))


@router.post("/forgot_password/", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(request: Request, serializer_write: ForgotPasswordSerializer) -> dict[str, str]:
    """Let the user request for a link to reset their password."""
    await serializer_write.run_async_validators(request=request)
    user = await serializer_write.update()
    return {"email": user.email}


@router.delete("/{pk}/", status_code=status.HTTP_204_NO_CONTENT)
async def destroy(pk: str) -> None:
    """Delete the authentication token used by the user."""
    instance = await User.find_one_or_404(User.auth_key == pk)
    await instance.set({"auth_key": None})
