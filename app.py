import click

from config import CONFIG
from data.data_generator import DataGenerator


@click.group()
def cli():
    pass


class DataGenerationConfig(object):
    def __init__(self):
        self.verbose = False
        self.model = False
        self.temperature = 0
        self.max_token = 0
        self.generator = None


data_gen_pass_config = click.make_pass_decorator(DataGenerationConfig, ensure=True)


@cli.group("generate")
@click.option("--verbose", is_flag=True)
@click.option("--out", type=click.File("w"), required=False)
@click.option("--max_token", default=CONFIG.MAX_TOKEN)
@click.option("--temperature", default=CONFIG.LLM_TEMPERATURE)
@click.option("--model", default="gpt-4o")
@data_gen_pass_config
def generate(config, model, temperature, max_token, out, verbose):
    config.model = model
    config.temperature = temperature
    config.verbose = verbose
    config.max_token = max_token
    config.output_file = out
    config.generator = DataGenerator(
        model=config.model,
        max_token=config.max_token,
        temperature=config.temperature,
        verbose=verbose,
    )


@generate.command("pictures_descriptions")
@click.option(
    "--pictures_dir", type=click.Path("w"), default=CONFIG.listing_pictures_dir
)
@data_gen_pass_config
def generate_pictures_descriptions(config, pictures_dir):
    config.generator.generate_pictures_descriptions(
        picture_dir=pictures_dir,
        output_file=config.output_file or CONFIG.listing_pictures_descr_file,
    )


@generate.command("listings")
@click.option(
    "--pictures_desc", type=click.Path("w"), default=CONFIG.listing_pictures_descr_file
)
@data_gen_pass_config
def generate_listings(config, pictures_desc):
    config.generator.generate_pictures_augmented_listings(
        picture_desc_file=pictures_desc,
        output_file=config.output_file or CONFIG.listing_file,
    )


if __name__ == "__main__":

    cli()
