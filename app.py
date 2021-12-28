"""Application entry point."""
from plotlyflask import init_app

appdash = init_app()
appdash.config["DEBUG"] = True

if __name__ == "__main__":
    appdash.run(host="0.0.0.0")
    