# 얼굴 인식 서버

이 프로젝트는 InsightFace 라이브러리를 사용한 실시간 얼굴 인식을 위한 Flask 기반 서버입니다. 비디오 스트림을 수신하고, 프레임을 처리하여 얼굴을 감지하고, 업로드된 대상 이미지와 비교하여 얼굴을 인식합니다. 서버는 라이브 비디오 피드와 인식된 얼굴 알림을 Flask-SocketIO를 사용하여 지원합니다.

## 기능

- 실시간 얼굴 감지 및 인식
- 업로드된 이미지를 인식할 대상 얼굴로 설정
- 클라이언트 장치에서 비디오 피드 스트림
- 얼굴이 인식되면 클라이언트에 알림
- 인식된 얼굴 이미지를 특정 디렉터리에 저장
- SSL을 사용한 보안 연결

## 사전 요구 사항

- Python 3.6 이상
- 필요한 Python 패키지 (`requirements.txt`에 나열됨)

## 설치

1. 레포지토리 클론:

    ```bash
    git clone https://github.com/your-repo/face-recognition-server.git
    cd face-recognition-server
    ```

2. 가상 환경 생성 및 활성화:

    ```bash
    python -m venv venv
    source venv/bin/activate  # 윈도우에서는 `venv\Scripts\activate`
    ```

3. 필요한 패키지 설치:

    ```bash
    pip install -r requirements.txt
    ```

4. 프로젝트 루트 디렉토리에 SSL 인증서(`cert.pem`)와 키(`key.pem`)를 배치합니다.

## 설정

- **업로드 폴더**: `static/uploads`
- **인식된 이미지 폴더**: `static/recognized`
- **대상 이미지**: `static/uploads/target.jpg`
- **TCP 서버**: `0.0.0.0:7777`
- **Flask 서버**: `0.0.0.0:5000`

## 서버 실행

서버를 실행하려면 다음 명령을 실행하십시오:

```bash
python app.py
