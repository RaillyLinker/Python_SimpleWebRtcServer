import cv2
import asyncio
import av
from fastapi import FastAPI, Request
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi.middleware.cors import CORSMiddleware

import filter

# FastAPI 객체 생성 및 CORS 설정(all open)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebRTC 연결 주소 매핑 (외부에서 http://localhost:8000/offer 이렇게 접속)
# 클라이언트(WebRTC Peer)로부터 offer SDP(Session Description Protocol)를 받아 WebRTC 연결을 시작 하는 엔드포인트
# 클라이언트는 /offer 의 request 에 JSON 으로 SDP 정보를 보냄
@app.post("/offer")
async def offer(request: Request):
    # 클라이언트에서 받은 offer SDP 를 저장
    data = await request.json()
    offer_sdp = data["sdp"]
    # Peer 연결 생성
    pc = RTCPeerConnection()

    # 데이터 수신부 콜백
    @pc.on("track")
    def on_track(track):
        # 영상 데이터를 받은 경우에 대한 처리
        if track.kind == "video":
            # 영상 데이터 프레임 처리 함수
            async def display_video():
                while True:
                    # track 에서 영상 프레임 받아 오기
                    frame: av.VideoFrame = await track.recv()
                    # 프레임 데이터 이미지 형식 변경
                    img = frame.to_ndarray(format="bgr24")  # OpenCV용으로 변환

                    img = filter.process_frame(img)  # 얼굴 이미지 덮기 처리

                    # OpenCV 창에 영상 img 표시
                    cv2.imshow("WebRTC Stream", img)
                    # OpenCV 창 닫기 조건
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()

            # 프레임 관련 처리 함수를 비동기로 실행
            asyncio.create_task(display_video())

    # 클라이언트의 offer SDP 를 수락
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp["sdp"], type=offer_sdp["type"]))
    # 서버에서 answer SDP 를 생성 후 설정
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 클라이언트에 서버의 answer SDP 를 응답으로 전송
    # 클라이언트는 이 정보를 통해 WebRTC 연결을 완료
    return {"sdp": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}}
