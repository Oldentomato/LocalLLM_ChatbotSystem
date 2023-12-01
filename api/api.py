from fastapi import FastAPI
# from fastapi_health import health
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

from router.use_model import usemodel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin에 대해 액세스 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드에 대해 액세스 허용
    allow_headers=["*"],  # 모든 헤더에 대해 액세스 허용
)

# 미들웨어: Trusted Host 설정 (보안을 위해 사용)
# app.add_middleware(
#     TrustedHostMiddleware, allowed_hosts=["localhost"]
# )

#로깅 설정
logging.basicConfig(level=logging.INFO)

app.include_router(usemodel, prefix="/model")

@app.get('/')
def home():
    return {'success': True}


