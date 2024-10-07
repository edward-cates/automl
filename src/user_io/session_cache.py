
from pydantic import BaseModel

class CacheObject(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    description: str
    value: object
    type_name: str

class SessionCache(BaseModel):
    """
    Cache for a single session.
    """

    cache: dict[str, CacheObject] = dict()

    def get(self, key: str) -> CacheObject:
        return self.cache[key]

    def get_value(self, key: str) -> object:
        return self.cache[key].value

    def set(self, key: str, value: CacheObject) -> None:
        self.cache[key] = value

    def describe(self) -> str:
        return "\n".join([
            f"{key}: {cache_obj.description} (value: {str(cache_obj.value)})"
            for key, cache_obj in self.cache.items()
        ])
