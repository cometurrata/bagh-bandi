import sys, os

sys.path.insert(0, os.getcwd())

from flask import Flask, send_from_directory, render_template
from src.domain.engine import Engine
from src.agents.minimax import MinimaxABAgent
import json
import flask

app = Flask(__name__, static_url_path='')

engine = Engine()

def buildResponse(msg = None):
    global engine
    ret = {}
    if msg is not None:
        ret ["msg"] = msg
    ret["goat_positions"] = engine.board.goatsPositions
    ret["tigers_positions"] = engine.board.tigerPositions
    ret["turn"] = engine.board.turn
    ret["re_capture_allowed"] = engine.re_capture_allowed
    response = flask.jsonify(ret)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    return ret

@app.route('/')
def root():
    return app.send_static_file("index.html")

agent = None

@app.route('/state', methods=['GET', 'POST'])
def state():
    return buildResponse()


@app.route('/move/<player_type>/<int:departure>/<int:destination>', methods=['GET', 'POST'])
def postId(player_type, departure, destination):
    global agent, engine
    msg = "OK"
    if player_type == engine.board.turn:
        if engine.is_valid_move(departure, destination, engine.board):
            engine.board = engine.move(departure, destination, engine.board)
            if agent and engine.board.turn != player_type:
                agent.moves()
            msg = "OK"
        else:
            msg = (f"/!\\ INVALID MOVE : {departure} to {destination}")
    else:
        msg = "NOT YOUR TURN"

    return buildResponse(msg)


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    global engine
    engine = Engine()
    msg = "OK"
    return buildResponse(msg)


@app.route('/skip/<player_type>', methods=['GET', 'POST'])
def skip_turn(player_type):
    global engine
    msg = "OK"
    if engine.re_capture_allowed and engine.board.turn == player_type and player_type == "tigers":
        engine.skip_tiger_recapture()
        if agent and engine.board.turn != player_type:
            agent.moves()
        msg =  "OK"
    else:
        msg = "Cannot skip turn"

    return buildResponse(msg)

@app.route('/setup_agent/<agent_type>', methods=['GET', 'POST'])
def setup_agent(agent_type):
    global agent
    agent = MinimaxABAgent(agent_type, engine)
    return buildResponse()

@app.route('/move_agent', methods=['GET', 'POST'])
def move_agent():
    global agent
    agent.moves()
    return buildResponse()


def main():
    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
