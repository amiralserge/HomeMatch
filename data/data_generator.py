import glob
import itertools
import os
from io import StringIO
from typing import Dict, List

import pandas as pd
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ratelimit import limits, sleep_and_retry

from config import CONFIG
from utils import local_image_to_data_url, split_in_chunks


class DataGenerator(object):
    def __init__(
        self,
        model=CONFIG.LLM_MODEL,
        max_token=CONFIG.MAX_TOKEN,
        temperature=CONFIG.LLM_TEMPERATURE,
        request_cool_down=int(CONFIG.LLM_REQUEST_COOLDOWN_TIME),
        verbose=False,
    ) -> None:
        self.llm_model = model
        self._max_token = max_token
        self._temperature = temperature
        self._request_cool_down = request_cool_down
        self._verbose = verbose
        self._llm = ChatOpenAI(
            model=model, max_tokens=max_token, temperature=temperature
        )

    def generate_pictures_descriptions(
        self,
        picture_dir=CONFIG.LISTING_PICTURES_DIR,
        output_file=CONFIG.LISTING_PICTURES_DESCR_FILE,
    ) -> None:
        data = []
        picture_collection = itertools.chain(
            glob.glob(f"./{picture_dir}/*.jpg"), glob.glob(f"./{picture_dir}/*.jpeg")
        )

        @sleep_and_retry
        @limits(calls=1, period=self._request_cool_down)
        def picture_desc(index, image_file):
            return dict(
                number=index,
                picture_file=image_file,
                image_desc=self._get_picture_description(image_file).replace(
                    "**", ""
                ),
            )

        for index, image_file in enumerate(picture_collection):
            data.append(picture_desc(index + 1, image_file))
        pd.DataFrame(data=data).to_csv(output_file)

    def _get_picture_description(self, picture_file):
        prompt = PromptTemplate.from_template(
            """{image}
Please describe the living room in the picture in terms of
Living Area, Kitchen, Flooring, Entrance, Additional Features, Lighting, Windows view, ceiling.
If a feature is missing ignore it.
if the room as a particular feature other than enumerated please describe it too.
Provide just the description as response"""
        )
        chain = ConversationChain(
            llm=self._llm,
            verbose=self._verbose,
        )
        return chain.run(prompt.format(image=local_image_to_data_url(picture_file)))

    def generate_pictures_augmented_listings(
        self,
        picture_desc_file: str = CONFIG.LISTING_PICTURES_DESCR_FILE,
        output_file: str = CONFIG.LISTING_FILE,
    ) -> None:

        if not os.path.isfile(picture_desc_file):
            raise Exception(
                f"Pictures desctiption files is none existant ({picture_desc_file})"
            )

        prompt = PromptTemplate(
            input_variables=["descriptions"],
            template="""
{descriptions}
-----------
INSTRUCTIONS: For each of real estates descriptions above Generate a distinct real estate listing following the example below.
In the description section, include description at your choice a description of amenities like gym, indoor/outdoor pool, backyard,
single/multiple cars parking garages, etc.
Each listing should have a different set of amenities.
A listing doesn't necessary need to have all the amenities.
The bedrooms number varies from 1 to 5.
EXAMPLE:
    Neighborhood: Green Oaks
    Price: $800,000
    Bedrooms: ?
    Bathrooms: ?
    House Size: ? sqft
    Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks.
    This charming ?-bedroom, ?-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure.
    Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes.
    The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family.
    Embrace sustainable living without compromising on style in this Green Oaks gem.
    Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens,
    and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe.
    With easy access to public transportation and bike lanes, commuting is a breeze.
OUTPUT FORMAT: return the result in a csv format with the headers number,neighborhood,price,bedrooms,bathrooms,house_size,description,neighborhood_description;
the content of every columns must be quoted except bedrooms and bathrooms.""",
        )
        columns = ["number", "picture_file", "image_desc"]
        picture_description_df = pd.read_csv(picture_desc_file)[columns]
        descriptions = picture_description_df.to_dict("records")

        def _process_response(llm_response):
            csv_buffer = StringIO(llm_response.replace("```csv", "").replace("```", ""))
            df = pd.read_csv(csv_buffer)
            # df.to_csv(f"{uuid.uuid4()}.csv")
            return df.to_dict("records")

        @sleep_and_retry
        @limits(calls=1, period=self._request_cool_down)
        def _llm_query(descriptions) -> List[Dict]:
            chain = ConversationChain(
                llm=self._llm,
                verbose=self._verbose,
            )
            response = chain.run(
                prompt.format(
                    descriptions="\n--------\n".join(
                        [
                            f"{d['number']}| {d['image_desc'].replace('**','')}"
                            for d in description_chunks
                        ]
                    ),
                )
            )
            return _process_response(response)

        listing_data = []
        for description_chunks in split_in_chunks(descriptions, 5):
            listing_data.extend(_llm_query(description_chunks))

        listing_df = pd.DataFrame(listing_data)
        # joining the listings df to the files
        listing_df = listing_df.join(
            picture_description_df.set_index("number"), on="number"
        )
        listing_df.drop(columns=["image_desc"], axis=1, inplace=True)
        listing_df.set_index("number")
        listing_df.to_csv(output_file)
