from lightning.pytorch.cli import LightningCLI
from src.lightning.lit_jepa import LitJEPA
from src.data.tiny_shakespeare import TinyShakespeareDataModule

def main():
    LightningCLI(LitJEPA, TinyShakespeareDataModule, save_config_callback=None)

if __name__ == "__main__":
    main()