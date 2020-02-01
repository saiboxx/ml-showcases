import numpy as np
from flask import render_template, request, Blueprint
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("title")
        axis.set_xlabel("x-axis")
        axis.set_ylabel("y-axis")
        axis.grid()
        axis.plot(range(5), range(5), "ro-")

        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)

        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(png_image.getvalue()).decode('utf8')

        return render_template("image.html", image=pngImageB64String)
    return render_template("index.html")
