import numpy as np
from flask import render_template, request, Blueprint, current_app
import io
import base64
import torch
from torchvision import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        decoder = current_app.config["MODEL"]
        with torch.no_grad():
            rand = torch.rand([25, 64])
            images = decoder(rand)

        plt.figure(figsize=(8, 5))
        grid = utils.make_grid(images, nrow=5)
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')

        # Convert plot to PNG image
        png_image = io.BytesIO()
        plt.savefig(png_image, format="png")

        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(png_image.getvalue()).decode('utf8')

        return render_template("index.html", image=pngImageB64String)
    return render_template("index.html", image=None)
