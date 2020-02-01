from flask import Flask
import torch
from .autoencoder import Decoder


def create_app():
    app = Flask(__name__)
    decoder = Decoder()
    decoder.load_state_dict(torch.load("models/decoder.pt"))
    decoder.eval()
    app.config['MODEL'] = decoder

    with app.app_context():
        from . import routes
        app.register_blueprint(routes.main_bp)

    return app
