# src/train.py
import sys
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
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
        
        def instantiate_classes(self) -> None:
            """Override to handle UnifiedDataModule's **kwargs gracefully."""
            # Let Lightning instantiate the trainer and model normally
            self.config_init = self.parser.instantiate_classes(self.config)
            
            # Access datamodule config and instantiate manually if needed
            # This bypasses strict validation for the datamodule kwargs
            self.datamodule = self.config_init.get("data")
            self.model = self.config_init.get("model") 
            self.trainer = self.config_init.get("trainer")

    cli = MyLightningCLI(
        LitJEPA, 
        UnifiedDataModule, 
        save_config_callback=None,
        auto_configure_optimizers=False
    )


if __name__ == "__main__":
    main()