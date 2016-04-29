from flask import Flask, jsonify, request, Response
from physics import simulation, predict_trajectory
import json
from rq import Queue
from werkzeug.contrib.cache import RedisCache
from worker import conn


app = Flask(__name__)
q = Queue(connection=conn)

cache = RedisCache(host='localhost', port=6379)

@app.route('/')
def home():
    cache.clear()
    q.empty()
    return jsonify({'valid':True})

@app.route('/start')
def start():
    cache.clear()
    q.empty()

    q.enqueue(simulation, timeout=1000)
    cache.set("running", True, 3600)
    return jsonify({'valid':True})

@app.route('/stop')
def stop():
    #running = not cache.get("running")
    cache.set("running", False, 3600)
    return jsonify({'valid':True})

@app.route('/predict')
def get_future():
    data = cache.get("data")
    trajectories = predict_trajectory(data)
    print trajectories
    return Response(json.dumps(trajectories), mimetype='application/json')
    
@app.route('/cache')
def get_cache():
    running = cache.get("running")
    data = cache.get("data")
    return jsonify({"data":data, "running":running})

@app.route('/status')
def get_status():
    return jsonify({"data":cache.get("running")})

@app.route('/thruster', methods=['GET'])
def fire_thruster():
    theta = request.GET.get("theta")
    if theta:
        cache.set("thruster_direction", theta, 3600)
    return jsonify({'valid':True})

if __name__ == '__main__':
    app.run(debug=True)
    
