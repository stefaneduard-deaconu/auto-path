from flask import Flask
from plugins.terrain import bp_terrain
from plugins.planner import bp_planner

app = Flask(__name__)

app.register_blueprint(bp_terrain)
app.register_blueprint(bp_planner)

