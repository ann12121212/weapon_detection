from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def send_detection_update(data):
    socketio.emit('detection_update', data)