# HomeMatch
HomeMatch is an llm powered personal real estate assistant.
## Installation
1. Create a virtual environment
```bash
python -m venv venv
source ./venv/bin/activate
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. configuration

Create a `.env` file for setting parameters. You can make a copy of the template
```bash
cp .env.template .env
```
The content of the .env must look as follows:
```
OPENAI_BASE_URL = https://api.openai.com/v1
OPENAI_API_KEY = here_goes_your_open_api_key
MAX_TOKENS = 2000
LLM_MODEL = gpt-4o-mini
LLM_TEMPERATURE = 0
LLM_REQUEST_COOLDOWN_TIME = 5
LISTING_PICTURES_DIR = ./listing_pictures
LISTING_PICTURES_DESCR_FILE = ./listing_pictures/pictures_descriptions.csv
LISTING_FILE = ./picture_augmented_listings.csv

VECTOR_DB_ENGINE = "lancedb"
VECTOR_DB_URI = ./homematch
```
These values will be the default unless specified otherwise in the `.env` file
## Usage
```bash
python app.y start
```
```bash
2024-08-24 23:49 HomeMatch INFO: HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
2024-08-24 23:49 HomeMatch INFO: HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
Running on local URL:  http://127.0.0.1:7860
2024-08-24 23:49 HomeMatch INFO: HTTP Request: GET http://127.0.0.1:7860/startup-events "HTTP/1.1 200 OK"
2024-08-24 23:49 HomeMatch INFO: HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
2024-08-24 23:49 HomeMatch INFO: HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
2024-08-24 23:49 HomeMatch INFO: HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
```
A gradio app should be avalable at http://127.0.0.1:7860