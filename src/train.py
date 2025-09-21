from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.lightning.lit_jepa import LitJEPA
from src.data.tiny_shakespeare import TinyShakespeareDataModule

def main():
    # 1. Instantiate the loggers manually
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="tb")
    wandb_logger = WandbLogger(
        project="unified-exit-jepa",
        name="tiny-jepa",
        save_dir="wandb_logs",
        log_model=False,
        tags=["tinyshakespeare", "jepa", "byol", "prototype"] # Now this works perfectly
    )

    # 2. Pass the instantiated loggers to the Trainer
    # We will subclass LightningCLI to inject our custom loggers
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            # This is a bit of a workaround to prevent the CLI from trying to
            # create its own logger. We "link" the logger argument to our list.
            parser.set_defaults({"trainer.logger": [tb_logger, wandb_logger]})

    # 3. Run the CLI
    cli = MyLightningCLI(LitJEPA, TinyShakespeareDataModule, save_config_callback=None)

if __name__ == "__main__":
    main()