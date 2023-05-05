from flask import Blueprint

bp_terrain = Blueprint('terrain', __name__)

@bp_terrain.route('/<page>')
def show(page):
    return 'terrain-ing...'