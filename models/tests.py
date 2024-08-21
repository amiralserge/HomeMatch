from models.listings import Listing, get_listing_summary


class TestListingModel:

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
        data = dict(
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
        listing = Listing(**data, listing_summary=get_listing_summary(data))

        assert listing.price == 650_000
        assert listing.bathrooms == 2
        assert listing.bedrooms == 3
        assert listing.house_size == 1800
        assert listing.description == (
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
        assert listing.neighborhood == "Sunset Heights"
        assert listing.neighborhood_description == (
            "Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and "
            "a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, "
            "and a community pool. The area offers convenient access to shopping, cafes, and schools, "
            "making it an ideal place for families and professionals alike."
        )
        assert (
            listing.listing_summary
            == """
Neighborhood: Sunset Heights
Price: $650,000
Bedrooms: 3
Bathrooms: 2
House Size: 1,800 sqft
Description: Welcome to Sunset Heights, where modern living meets comfort! This stunning 3-bedroom, 2-bathroom home features an inviting living area adorned with a cozy sofa and a contemporary coffee table, perfect for gatherings. The kitchen boasts high-end appliances and a stylish island that provides both prep space and casual dining. Enjoy the luxury of a backyard oasis complete with a patio for entertaining and a beautifully landscaped garden. Additional amenities include a two-car garage and access to a community gym, ensuring a healthy lifestyle is at your fingertips.
Neighborhood Description: Sunset Heights is a vibrant neighborhood known for its family-friendly atmosphere and a wealth of outdoor recreational options. Residents enjoy local parks, hiking trails, and a community pool. The area offers convenient access to shopping, cafes, and schools, making it an ideal place for families and professionals alike.
"""
        )  # noqa: E501
