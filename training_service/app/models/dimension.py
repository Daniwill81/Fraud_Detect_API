"""
Dimension.

Dimension represents.
"""
from sap.beanie import Document


class Dimension(Document):
    """A number of NN in a layer."""

    layer_one: int
    hidden_layer_one: int
    hidden_layer_two: int
    hidden_layer_three: int
    final_layer: int

    class Settings:
        """Settings for the database collection."""

        name = "dimension"
