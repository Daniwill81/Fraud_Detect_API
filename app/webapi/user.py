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

from sap.fastapi.pagination import CursorInfo, PaginatedData

from app.models import User
from app.models.enums import RoleEnum
from app.models.user.auth import user_auth
from app.serializers.user import UserSerializer, WriteUserSerializer

router = APIRouter()


@router.get("/current/", status_code=status.HTTP_200_OK)
async def current(request_user: User = Depends(user_auth.authenticate)) -> UserSerializer:
    """Retrieve the currently authenticated user."""
    return UserSerializer.read(request_user)


@router.get("/{pk}/", status_code=status.HTTP_200_OK)
async def retrieve(
    pk: str,
) -> UserSerializer:
    """Retrieve a user by id."""
    instance = await User.get_or_404(pk)
    return UserSerializer.read(instance)


@router.get("/", status_code=status.HTTP_200_OK)
async def listing(request: Request) -> PaginatedData[UserSerializer]:
    """Retrieve all agencies.

    This is a public endpoint. Anyone to list info about all agency.
    """

    cursor = CursorInfo(request=request)

    qs = User.find()
    instance_list = await qs.find(**cursor.get_beanie_query_params()).to_list()

    cursor.set_count(await qs.count())
    result: PaginatedData[UserSerializer] = UserSerializer.read_page(instance_list, request=request, cursor_info=cursor)

    return result


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create(
    request: Request,
    serializer_write: WriteUserSerializer,
    # request_user: User = Depends(user_auth.require(RoleEnum.get_super_admin())),
) -> UserSerializer:
    """Create a user."""
    await serializer_write.run_async_validators(request=request)
    instance = await serializer_write.create(request=request)  # , request_user=request_user
    return UserSerializer.read(instance)


@router.put("/{pk}/", status_code=status.HTTP_202_ACCEPTED)
async def update(
    request: Request,
    pk: str,
    serializer_write: WriteUserSerializer,
    request_user: User = Depends(user_auth.require(RoleEnum.get_list_all_authenticated())),
) -> UserSerializer:
    """Update an user."""
    serializer_write.instance = await User.get_or_404(pk)
    await serializer_write.run_async_validators(request=request)
    instance = await serializer_write.update(request=request, request_user=request_user)
    return UserSerializer.read(instance)
