import uuid
from typing import Dict

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import AliasGenerator, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

_openapi = get_registry().get("openai").create()


class Listing(LanceModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda name: name.lower().replace(" ", "_"),
            serialization_alias=lambda name: " ".join(
                word.capitalize() for word in name.split("_")
            ),
        )
    )
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    vector: Vector(_openapi.ndims()) = None  # type: ignore
    image_vector: Vector(512) = None  # type: ignore
    image: bytes = None
    neighborhood: str
    price: float
    bedrooms: int
    bathrooms: int
    house_size: float
    description: str
    neighborhood_description: str
    listing_summary: str = None

    @field_validator("house_size", mode="before")
    def parse_house_size(cls, value):
        if value and isinstance(value, str):
            return float(value.replace("sqft", "").replace(",", "").strip())
        return value

    @field_validator("price", mode="before")
    def parse_price(cls, value):
        if value and isinstance(value, str):
            return float(value.replace("$", "").replace(",", ""))
        return value

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.listing_summary:
            self.listing_summary = get_listing_summary(self)


def get_listing_summary(data: Dict | Listing) -> str:
    if isinstance(data, Listing):
        fields = [
            "neighborhood",
            "price",
            "bedrooms",
            "bathrooms",
            "house_size",
            "description",
            "neighborhood_description",
        ]
        data = data.model_dump(include=fields)
    neighborhood = data.get("Neighborhood") or data.get("neighborhood")
    price = Listing.parse_price(data.get("Price") or data.get("price") or "")
    price = f"{price:,.0f}"
    bedrooms = data.get("Bedrooms") or data.get("bedrooms")
    bathrooms = data.get("Bathrooms") or data.get("bathrooms")
    size = Listing.parse_house_size(
        (data.get("House Size") or data.get("house_size") or "")
    )
    size = f"{size:,.2f}"
    description = data.get("Description") or data.get("description")
    neighborhood_description = data.get("Neighborhood Description") or data.get(
        "neighborhood_description"
    )
    return f"""
Neighborhood: {neighborhood}
Price: ${price}
Bedrooms: {bedrooms}
Bathrooms: {bathrooms}
House Size: {size} sqft
Description: {description}
Neighborhood Description: {neighborhood_description}
"""
