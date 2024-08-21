from models.listings import Listing, get_listing_summary


class TestListingModel:

    @classmethod
    def _create_listing_record(cls, data):
        return Listing(**data, listing_summary=get_listing_summary(data))

    def test_get_listing_description(self):
        listing_data = dict(
            price="650,000",
            bathrooms=2,
            bedrooms=3,
            house_size="1,800 sqft",
            description=(
                "Welcome to Sunset Heights, where modern living meets comfort! "
                "This stunning 3-bedroom, 2-bathroom home features an inviting living "
                "area adorned with a cozy sofa and a contemporary coffee table, "
                "perfect for gatherings. The kitchen boasts high-end appliances "
                "and a stylish island that provides both prep space and casual "
                "dining. Enjoy the luxury of a backyard oasis complete with a patio for "
                "entertaining and a beautifully landscaped garden. "
                "Additional amenities include a two-car garage and access to a community gym, "
                "ensuring a healthy lifestyle is at your fingertips."
            ),
            neighborhood="Sunset Heights",
            neighborhood_description=(
                "Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and "
                "a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, "
                "and a community pool. The area offers convenient access to shopping, cafes, and schools, "
                "making it an ideal place for families and professionals alike."
            ),
        )
        expected = """
Neighborhood: Sunset Heights
Price: $650,000
Bedrooms: 3
Bathrooms: 2
House Size: 1,800 sqft
Description: Welcome to Sunset Heights, where modern living meets comfort! This stunning 3-bedroom, 2-bathroom home features an inviting living area adorned with a cozy sofa and a contemporary coffee table, perfect for gatherings. The kitchen boasts high-end appliances and a stylish island that provides both prep space and casual dining. Enjoy the luxury of a backyard oasis complete with a patio for entertaining and a beautifully landscaped garden. Additional amenities include a two-car garage and access to a community gym, ensuring a healthy lifestyle is at your fingertips.
Neighborhood Description: Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, and a community pool. The area offers convenient access to shopping, cafes, and schools, making it an ideal place for families and professionals alike.
"""  # noqa: E501
        assert get_listing_summary(listing_data) == expected

    def test_listing_model(self):
        listing_record1 = self._create_listing_record(
            dict(
                price="$650,000",
                bathrooms=2,
                bedrooms=3,
                house_size="1,800 sqft",
                description=(
                    "Welcome to Sunset Heights, where modern living meets comfort! "
                    "This stunning 3-bedroom, 2-bathroom home features an inviting living "
                    "area adorned with a cozy sofa and a contemporary coffee table, "
                    "perfect for gatherings. The kitchen boasts high-end appliances "
                    "and a stylish island that provides both prep space and casual "
                    "dining. Enjoy the luxury of a backyard oasis complete with a patio for "
                    "entertaining and a beautifully landscaped garden. "
                    "Additional amenities include a two-car garage and access to a community gym, "
                    "ensuring a healthy lifestyle is at your fingertips."
                ),
                neighborhood="Sunset Heights",
                neighborhood_description=(
                    "Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and "
                    "a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, "
                    "and a community pool. The area offers convenient access to shopping, cafes, and schools, "
                    "making it an ideal place for families and professionals alike."
                ),
            )
        )

        assert listing_record1.price == 650_000
        assert listing_record1.bathrooms == 2
        assert listing_record1.bedrooms == 3
        assert listing_record1.house_size == 1800
        assert listing_record1.description == (
            "Welcome to Sunset Heights, where modern living meets comfort! "
            "This stunning 3-bedroom, 2-bathroom home features an inviting living "
            "area adorned with a cozy sofa and a contemporary coffee table, "
            "perfect for gatherings. The kitchen boasts high-end appliances "
            "and a stylish island that provides both prep space and casual "
            "dining. Enjoy the luxury of a backyard oasis complete with a patio for "
            "entertaining and a beautifully landscaped garden. "
            "Additional amenities include a two-car garage and access to a community gym, "
            "ensuring a healthy lifestyle is at your fingertips."
        )
        assert listing_record1.neighborhood == "Sunset Heights"
        assert listing_record1.neighborhood_description == (
            "Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and "
            "a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, "
            "and a community pool. The area offers convenient access to shopping, cafes, and schools, "
            "making it an ideal place for families and professionals alike."
        )

        assert listing_record1.listing_summary == (
            """
Neighborhood: Sunset Heights
Price: $650,000
Bedrooms: 3
Bathrooms: 2
House Size: 1,800 sqft
Description: Welcome to Sunset Heights, where modern living meets comfort! This stunning 3-bedroom, 2-bathroom home features an inviting living area adorned with a cozy sofa and a contemporary coffee table, perfect for gatherings. The kitchen boasts high-end appliances and a stylish island that provides both prep space and casual dining. Enjoy the luxury of a backyard oasis complete with a patio for entertaining and a beautifully landscaped garden. Additional amenities include a two-car garage and access to a community gym, ensuring a healthy lifestyle is at your fingertips.
Neighborhood Description: Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, and a community pool. The area offers convenient access to shopping, cafes, and schools, making it an ideal place for families and professionals alike.
"""
        )  # noqa: E501, E122

        assert listing_record1.model_dump(
            by_alias=True,
            exclude=["id", "image", "image_vector", "listing_summary", "vector"],
        ) == {
            "Price": 650_000,
            "Bathrooms": 2,
            "Bedrooms": 3,
            "House Size": 1_800,
            "Description": (
                "Welcome to Sunset Heights, where modern living meets comfort! "
                "This stunning 3-bedroom, 2-bathroom home features an inviting living "
                "area adorned with a cozy sofa and a contemporary coffee table, "
                "perfect for gatherings. The kitchen boasts high-end appliances "
                "and a stylish island that provides both prep space and casual "
                "dining. Enjoy the luxury of a backyard oasis complete with a patio for "
                "entertaining and a beautifully landscaped garden. "
                "Additional amenities include a two-car garage and access to a community gym, "
                "ensuring a healthy lifestyle is at your fingertips."
            ),
            "Neighborhood": "Sunset Heights",
            "Neighborhood Description": (
                "Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and "
                "a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, "
                "and a community pool. The area offers convenient access to shopping, cafes, and schools, "
                "making it an ideal place for families and professionals alike."
            ),
        }
