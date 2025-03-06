"""
Enums.

Platforme users role enumeration.
"""

import enum


class RoleEnum(str, enum.Enum):
    """List of roles available for users.

    Each role has different access and set of permissions.
    """

    ADMIN = " SUPER ADMIN"  # Platform - Super-administrateur
    INST = "INSTITUTION"  # Financial institution - user

    @classmethod
    def get_super_admin(cls) -> list["RoleEnum"]:
        """Get the super users."""
        return [
            cls.ADMIN,
        ]

    @classmethod
    def get_authenticated(cls) -> list["RoleEnum"]:
        """Get the institution authenticaded users."""
        return [
            cls.INST,
        ]

    @classmethod
    def get_list_all_authenticated(cls) -> list["RoleEnum"]:
        """Return a list of roles that are authenticated."""
        return cls.get_super_admin() + [cls.INST]


class SexEnum(str, enum.Enum):
    """Sex of users"""

    M = "Male"
    F = "Female"
