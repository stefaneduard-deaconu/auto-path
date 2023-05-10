from flask import Blueprint
from tasks.terrain import task_terrain_from_config

bp_terrain = Blueprint('terrain', __name__)

@bp_terrain.route('/<page>')
def show(page):
    return 'terrain-ing...'