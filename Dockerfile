# 베이스 이미지로 Python과 Node.js를 모두 포함한 이미지 사용
FROM continuumio/miniconda3 AS base

# 작업 디렉토리 설정
WORKDIR /app

# Node.js 환경 설정
FROM node:14-alpine AS node_deps

# Node.js 의존성 설치
WORKDIR /app
COPY package*.json ./
RUN npm install --production

# Python 환경 설정
FROM base AS python_deps

# conda 가상환경 생성 및 필요한 패키지 설치
COPY environment.yml .
RUN conda env create -f environment.yml

# 가상환경 활성화
SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]

# 애플리케이션 소스 복사
COPY . .

# Python, Node.js 모두 설치 완료된 최종 이미지
FROM base

# Node.js 및 Python 의존성 복사
COPY --from=node_deps /app /app
COPY --from=python_deps /opt/conda /opt/conda

# 가상환경 활성화 스크립트 복사
COPY --from=python_deps /opt/conda/envs/venv /opt/conda/envs/venv

# 환경 변수 설정
ENV PATH=/opt/conda/envs/venv/bin:$PATH

# Node.js와 Python 실행 포트 설정
EXPOSE 8080

# 서버 시작 명령어 (Node.js 서버)
CMD ["npm", "start"]
