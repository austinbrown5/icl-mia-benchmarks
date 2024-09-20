class Prompt:
    def __init__(self) -> None:
        self.prompts = {
            "general": (
                "INSTRUCTION:\n"
                "You are provided with the FIRST PIECE of a question from the train split of a dataset.\n"
                "Finish the SECOND PIECE of the question as exactly appeared in the dataset.\n"
                "Only rely on the original form of the question in the dataset to finish the SECOND PIECE.\n\n"
                "FIRST PIECE:\n"
                "{first_piece}\n\n"
                "SECOND PIECE:\n"
            ),
            "guided": (
                "Instruction:\n"
                "You are provided with the FIRST PIECE of a question from the train split of the {dataset_name} dataset.\n"
                "Finish the SECOND PIECE of the question as exactly appeared in the dataset.\n"
                "Only rely on the original form of the question in the dataset to finish the SECOND PIECE.\n\n"
                "FIRST PIECE:\n"
                "{first_piece}\n\n"
                "SECOND PIECE:\n"
            )
        }

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")