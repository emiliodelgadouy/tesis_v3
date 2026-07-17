from src.model_builder.simple import SimpleModelBuilder


class FullModelBuilder(SimpleModelBuilder):
    # igual q simple pero con IMG_SIZE fijo 512x512 (se pasa desde el notebook)
    model_name = "full"
