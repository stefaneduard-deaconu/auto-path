from flask import Blueprint

bp_planner = Blueprint('planner', __name__)

@bp_planner.route('/<page>')
def show(page):
    return 'planning...'