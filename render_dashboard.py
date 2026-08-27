from pathlib import Path
from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("app/templates")
)

template = env.get_template("dashboard.html")
html = template.render()

Path("deploy/index.html").write_text(
    html,
    encoding="utf-8"
)

print("Created deploy/index.html")
