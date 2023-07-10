from enum import EnumMeta

from strenum import StrEnum as StrEnumBase


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        return item in tuple(cls)


class StrEnum(StrEnumBase, metaclass=MetaEnum):
    ...
