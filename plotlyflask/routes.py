"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="Chứng khoáng",
        description="Visualiz lên webapp.",
        template="home-template",
        body="This is a homepage served with Flask.",
    )