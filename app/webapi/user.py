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

from fastapi import APIRouter, Depends, Request, status

from sap.beanie.query import prefetch_related
from sap.fastapi.pagination import CursorInfo, PaginatedData

from app import controllers
from app.models import User
from app.models.enums import RoleEnum
from app.models.user.auth import user_auth
from app.query.user import UserQuery
from app.serializers.user import APIKeySerializer, UserSerializer, WriteUserSerializer

router = APIRouter()


@router.get("/current/", status_code=status.HTTP_200_OK)
async def current(request_user: User = Depends(user_auth.authenticate)) -> UserSerializer:
    """Retrieve the currently authenticated user."""
    return UserSerializer.read(request_user)


@router.get("/{pk}/", status_code=status.HTTP_200_OK)
async def retrieve(
    pk: str,
    # request_user: User = Depends(user_auth.require([RoleEnum.DSI1])),
) -> UserSerializer:
    """Retrieve a user by id."""
    instance = await User.get_or_404(pk)
    return UserSerializer.read(instance)


@router.get("/", status_code=status.HTTP_200_OK)
async def listing(
    request: Request,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_primary())),
) -> PaginatedData[UserSerializer]:
    """Retrieve all user."""
    cursor = CursorInfo(request=request)
    query = UserQuery(user=request_user, filters=request.query_params)

    if search_text := request.query_params.get("q"):
        qs = query.get_search(search_text)
    else:
        qs = query.get_qs().find(**cursor.get_beanie_query_params())

    instance_list = await qs.to_list()
    await prefetch_related(instance_list, to_attribute="organization")

    cursor.set_count(await qs.count())
    result: PaginatedData[UserSerializer] = UserSerializer.read_page(
        instance_list,
        request=request,
        cursor_info=cursor,
    )
    return result


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create(
    request: Request,
    serializer_write: WriteUserSerializer,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_primary())),
) -> UserSerializer:
    """Create a user."""
    await serializer_write.run_async_validators(request=request)
    instance = await serializer_write.create(request=request, request_user=request_user)
    return UserSerializer.read(instance)


@router.put("/{pk}/", status_code=status.HTTP_202_ACCEPTED)
async def update(
    request: Request,
    pk: str,
    serializer_write: WriteUserSerializer,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_primary())),
) -> UserSerializer:
    """Update an user."""
    serializer_write.instance = await User.get_or_404(pk)
    await serializer_write.run_async_validators(request=request)
    instance = await serializer_write.update(request=request, request_user=request_user)
    return UserSerializer.read(instance)


@router.post("/{pk}/user_activate/", status_code=status.HTTP_202_ACCEPTED)
async def user_activate(
    pk: str,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_primary())),
) -> UserSerializer:
    """Perform action: Activate user."""
    instance = await User.get_or_404(pk)
    await controllers.user.user_activate(instance)
    return UserSerializer.read(instance)


@router.post("/{pk}/user_deactivate/", status_code=status.HTTP_202_ACCEPTED)
async def user_deactivate(
    pk: str,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_primary())),
) -> UserSerializer:
    """Perform action: Deactivate user."""
    instance = await User.get_or_404(pk)
    await controllers.user.user_deactivate(instance)
    return UserSerializer.read(instance)


@router.get("/current/api_key/", status_code=status.HTTP_200_OK)
async def current_api_key(
    request_user: User = Depends(user_auth.require([RoleEnum.BANK1, RoleEnum.ASIN1]))
) -> APIKeySerializer:
    """Retrieve the currently authenticated user's API Key."""
    if not request_user.api_key:
        await request_user.generate_api_key()
    return APIKeySerializer.read(request_user)
