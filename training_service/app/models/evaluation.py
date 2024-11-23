"""
Evaluation.

Evaluation represents.
"""
from sap.beanie import Document


class Evaluation(Document):
    """A number of NN in a layer."""

    log_loss: int
    accurancy: int

    class Settings:
        """Settings for the database collection."""

        name = "evaluation"
