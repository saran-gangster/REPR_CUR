from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from src.lightning.lit_jepa import LitJEPA
from src.data.unified import UnifiedDataModule


def main():
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="tb")
    wandb_logger = WandbLogger(
        project="unified-exit-jepa",
        name=None,  # let W&B auto-name or set via CLI: --trainer.logger.init_args.name="run-name"
        save_dir="wandb_logs",
        log_model=False,
        tags=["jepa", "byol", "prototype"],
    )

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"trainer.logger": [tb_logger, wandb_logger]})

    cli = MyLightningCLI(LitJEPA, UnifiedDataModule, save_config_callback=None)


if __name__ == "__main__":
    main()