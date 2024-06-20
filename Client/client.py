import socket
import pyautogui
import numpy as np
import cv2
import time

TCP_IP = "34.47.80.221"  # 서버의 IP 주소
TCP_PORT = 7777  # 서버의 포트 번호

def connect_to_server():
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((TCP_IP, TCP_PORT))
            print("Connected to server at {}:{}".format(TCP_IP, TCP_PORT))
            return client_socket
        except Exception as e:
            print("Connection failed, retrying in 5 seconds:", e)
            time.sleep(5)

def send_video_frames(client_socket):
    left, top, width, height = 0, 0, 1920, 1080
    
    while True:
        try:
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (480, 540))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            size = len(frame_bytes)
            client_socket.sendall(size.to_bytes(4, byteorder='big'))
            client_socket.sendall(frame_bytes)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except socket.error as e:
            print("Socket error, reconnecting:", e)
            client_socket.close()
            client_socket = connect_to_server()
        except Exception as e:
            print("Error:", e)
            break

    client_socket.close()

if __name__ == '__main__':
    client_socket = connect_to_server()
    send_video_frames(client_socket)
