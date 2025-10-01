# src/train.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from src.lightning.lit_jepa import LitJEPA
from src.data.unified import UnifiedDataModule


def main():
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="tb")
    wandb_logger = WandbLogger(
        project="unified-exit-jepa",
        name=None,
        save_dir="wandb_logs",
        log_model=False,
        tags=["jepa", "byol", "prototype"],
    )

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"trainer.logger": [tb_logger, wandb_logger]})
            # Allow arbitrary kwargs for UnifiedDataModule by disabling strict validation
            # This is needed because UnifiedDataModule forwards **kwargs to the underlying datamodule
            parser.set_defaults({"data": {}})

    cli = MyLightningCLI(
        LitJEPA, 
        UnifiedDataModule, 
        save_config_callback=None,
        parser_kwargs={
            "fit": {"default_config_files": []},
            "parser_mode": "omegaconf"  # Use OmegaConf mode for more flexible config handling
        }
    )


if __name__ == "__main__":
    main()