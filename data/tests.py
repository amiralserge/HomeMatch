import io
from unittest import mock

import pytest

from data.data_generator import DataGenerator


class TestDataGenerator:

    @mock.patch("data.data_generator.glob.glob")
    def test_generate_pictutes_description(self, mock_glob):

        def glob_side_effect(*args):
            if args[0].endswith("*.jpg"):
                return ["path/to/img1.jpg", "path/to/img2.jpg"]
            if args[0].endswith("*.jpeg"):
                return ["path/to/img3.jpeg", "path/to/img4.jpeg"]

        mock_glob.side_effect = glob_side_effect

        with mock.patch.object(
            DataGenerator, "_get_llm_picture_description"
        ) as mock_llm_picture_description:
            mock_llm_picture_description.side_effect = (
                lambda picture_file: f"description for {picture_file}"
            )
            generator = DataGenerator(request_cool_down=0)

            output_file = io.StringIO()
            generator.generate_pictures_descriptions(
                picture_dir="./picture_dir", output_file=output_file
            )

        assert output_file.getvalue().splitlines() == [
            "number,picture_file,image_desc",
            "1,path/to/img1.jpg,description for path/to/img1.jpg",
            "2,path/to/img2.jpg,description for path/to/img2.jpg",
            "3,path/to/img3.jpeg,description for path/to/img3.jpeg",
            "4,path/to/img4.jpeg,description for path/to/img4.jpeg",
        ]

    def test_generate_listing_description(self):
        generator = DataGenerator(request_cool_down=0)

        with pytest.raises(DataGenerator.NonExistentFileException):
            generator.generate_pictures_augmented_listings(
                picture_desc_file="non_existing_file", output_file="output.csv"
            )

        with mock.patch("data.data_generator.os.path.isfile"), mock.patch.object(
            DataGenerator, "_generate_listings_with_llm"
        ) as mock_generate_listings_with_llm:

            def _generate_listings_with_llm(picture_descriptions):
                csv_header = "number,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description"
                csv_values = "\n".join(
                    [
                        f"{picture_descr['number']},neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description"
                        for picture_descr in picture_descriptions
                    ]
                )
                return f"```csv\n{csv_header}\n{csv_values}```"

            mock_generate_listings_with_llm.side_effect = _generate_listings_with_llm
            picture_desc_file = io.StringIO(
                """number,picture_file,image_desc
1,path/to/img1.jpg,"description for path/to/img1.jpg"
2,path/to/img2.jpg,"description for path/to/img2.jpg"
3,path/to/img3.jpeg,"description for path/to/img3.jpeg"
4,path/to/img4.jpeg,"description for path/to/img4.jpeg"
5,path/to/img5.jpeg,"description for path/to/img5.jpeg"
6,path/to/img6.jpeg,"description for path/to/img6.jpeg"
7,path/to/img7.jpeg,"description for path/to/img7.jpeg"
8,path/to/img8.jpeg,"description for path/to/img8.jpeg"
"""
            )
            output_file = io.StringIO()

            generator.generate_pictures_augmented_listings(
                picture_desc_file=picture_desc_file, output_file=output_file
            )

            assert mock_generate_listings_with_llm.call_count == 2
            call1 = mock.call(
                [
                    {
                        "number": 1,
                        "picture_file": "path/to/img1.jpg",
                        "image_desc": "description for path/to/img1.jpg",
                    },
                    {
                        "number": 2,
                        "picture_file": "path/to/img2.jpg",
                        "image_desc": "description for path/to/img2.jpg",
                    },
                    {
                        "number": 3,
                        "picture_file": "path/to/img3.jpeg",
                        "image_desc": "description for path/to/img3.jpeg",
                    },
                    {
                        "number": 4,
                        "picture_file": "path/to/img4.jpeg",
                        "image_desc": "description for path/to/img4.jpeg",
                    },
                    {
                        "number": 5,
                        "picture_file": "path/to/img5.jpeg",
                        "image_desc": "description for path/to/img5.jpeg",
                    },
                ]
            )
            call2 = mock.call(
                [
                    {
                        "number": 6,
                        "picture_file": "path/to/img6.jpeg",
                        "image_desc": "description for path/to/img6.jpeg",
                    },
                    {
                        "number": 7,
                        "picture_file": "path/to/img7.jpeg",
                        "image_desc": "description for path/to/img7.jpeg",
                    },
                    {
                        "number": 8,
                        "picture_file": "path/to/img8.jpeg",
                        "image_desc": "description for path/to/img8.jpeg",
                    },
                ]
            )
            mock_generate_listings_with_llm.assert_has_calls([call1, call2])
        assert output_file.getvalue().splitlines() == [
            "number,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,picture_file",
            "1,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img1.jpg",
            "2,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img2.jpg",
            "3,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img3.jpeg",
            "4,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img4.jpeg",
            "5,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img5.jpeg",
            "6,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img6.jpeg",
            "7,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img7.jpeg",
            "8,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description,path/to/img8.jpeg",
        ]
