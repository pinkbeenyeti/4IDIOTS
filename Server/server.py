from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import socket
import threading
import ssl
import os
from insightface.app import FaceAnalysis
import concurrent.futures
import numpy as np
import cv2
from flask_socketio import SocketIO, emit

# 서버 설정
app = Flask(__name__, template_folder='my_template')
socketio = SocketIO(app)

# 통신 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_IP = '0.0.0.0'
TCP_PORT = 7777
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen(3)

# 업로드 사진 파일 설정
UPLOAD_FOLDER = 'static/uploads'
CHECK_FILE = 'static/uploads/target.jpg'
RECOGNIZED_FOLDER = 'static/recognized'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECOGNIZED_FOLDER'] = RECOGNIZED_FOLDER

# 인식 모델 설정
model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))
state = 0

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img)
    if len(faces) == 0:
        raise ValueError("No faces detected")
    embeddings = [face.normed_embedding for face in faces]
    return embeddings, faces

# 코사인 유사도 계산 함수
def calculate_cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# 인식 모델 실행 여부 확인 및 실행함수
def execute_model(frame_count, frame):
    # 매 60프레임마다 처리 (1초 단위)
    if frame_count % 60 == 0 and state == 1:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = model.get(frame_rgb)

        highest_similarity = -1
        best_match_face = None
        
        for face in faces:
            embedding2 = face.normed_embedding
            similarity = calculate_cosine_similarity(embedding1, embedding2)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_face = face

        if best_match_face is not None and highest_similarity >= similarity_threshold:
            box = best_match_face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, f'Similarity: {highest_similarity:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # 인식된 프레임 저장
            recognized_path = os.path.join(app.config['RECOGNIZED_FOLDER'], 'recognized.jpg')
            cv2.imwrite(recognized_path, frame)
            
            # 인식된 이미지를 클라이언트에 알림
            socketio.emit('image_recognized', {'file_path': 'recognized.jpg'})  # 변경된 부분

# 영상 수신 및 출력
def generate_frames(client_socket):
    frame_count = 0
    while True:
        try:
            # 데이터 크기 수신
            data_size = client_socket.recv(4)
            if not data_size:
                break
            size = int.from_bytes(data_size, byteorder='big')
            
            # 데이터 수신
            frame_bytes = b''
            while len(frame_bytes) < size:
                packet = client_socket.recv(size - len(frame_bytes))
                if not packet:
                    break
                frame_bytes += packet
            
            # 수신한 데이터를 이미지로 디코딩
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # 인식 모델 수행
            frame_count += 1
            execute_model(frame_count, frame)
            
            # 이미지를 JPEG 형식으로 인코딩하여 전송
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 이미지 스트림 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print("Error:", e)
            break

@app.route('/')
def index():
    return render_template('index2.html', uploaded_image_url=None)

@app.route('/video_feed')
def video_feed():
    print("Waiting for connection...")
    client_socket, addr = server_socket.accept()
    print("Connected to", addr)
    return Response(generate_frames(client_socket), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognized_feed')
def recognized_feed():
    return send_from_directory(app.config['RECOGNIZED_FOLDER'], 'recognized.jpg')

def save_uploaded_file(file):
    if file.filename == '':
        return None
    if file:
        global state, embedding1, similarity_threshold
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'target.jpg')
        file.save(file_path)
        
        # 이미지 전처리
        embedding1, faces1 = preprocess_image(CHECK_FILE)
        
        # image_path의 첫 번째 얼굴 임베딩 사용
        embedding1 = embedding1[0]
        similarity_threshold = 0.23  # 임계값 설정 (예: 0.3)
        state = 1
        
        socketio.emit('image_uploaded', {'file_path': 'target.jpg'})
        return file_path
    return None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    # 파일 저장을 비동기적으로 처리
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(save_uploaded_file, file)
        file_path = future.result()
        
    if file_path:
        return render_template('index2.html', uploaded_image_url=file_path)
    else:
        return redirect(request.url)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def run_flask():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    ssl_context.load_cert_chain(certfile='./cert.pem', keyfile='./key.pem')
    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)

def run_socket_server():
    threading.Thread(target=run_flask).start()

if __name__ == '__main__':
    run_socket_server()
