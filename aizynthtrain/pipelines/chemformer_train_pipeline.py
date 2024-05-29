"""Module containing training a synthesis Chemformer model"""
import os
import subprocess

from metaflow import FlowSpec, Parameter, step

from aizynthtrain.utils.configs import ChemformerTrainConfig, load_config


class ChemformerTrainFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: ChemformerTrainConfig = load_config(
            self.config_path, "chemformer_train"
        )
        self.next(self.train_model)

    @step
    def train_model(self):
        """Running Chemformer fine-tuning using the chemformer environment"""
        args = self._prompt_from_config()

        cmd = f"python -m molbart.fine_tune {args}"

        if "CHEMFORMER_ENV" in os.environ:
            cmd = f"conda run -p {os.environ['CHEMFORMER_ENV']} " + cmd

            if "CONDA_PATH" in os.environ:
                cmd = os.environ["CONDA_PATH"] + cmd

        subprocess.check_call(cmd, shell=True)
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Fine-tuned Chemformer model located here: {self.config.output_directory}"
        )

    def _prompt_from_config(self) -> str:
        """Get argument string for running fine-tuning python script from config."""
        args = []
        for param in self.config:
            if param[0] == "model_path":
                value = param[1].replace("=", "\=")
                args.append(f"'{param[0]}={value}'")
                continue
            args.append(f"'{param[0]}={param[1]}'")

        prompt = " ".join(args)
        return prompt


if __name__ == "__main__":
    ChemformerTrainFlow()
