# # 베이스 이미지로 Python과 Node.js를 모두 포함한 이미지 사용
# FROM continuumio/miniconda3:4.10.3-alpine AS python_deps

# # 필요한 패키지 설치
# RUN apk add --no-cache bash

# # 작업 디렉토리 설정
# WORKDIR /app
# # Conda 업데이트 (최신 버전 패키지를 설치하기 위함)
# # RUN conda update -n base -c defaults conda
# # Conda 업데이트 및 mamba 설치
# RUN /opt/conda/bin/conda update -n base -c defaults conda && \
#     /opt/conda/bin/conda install mamba -n base -c conda-forge

# # environment.yml 복사 및 환경 생성
# COPY environment.yml .
# RUN conda env create -f environment.yml || conda clean --all --yes

# # 가상환경 활성화
# SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]

# # 가상환경의 Python과 다른 파일들을 복사
# # 기본 경로에서 venv의 Python을 사용할 수 있도록 복사
# RUN mkdir -p /app/venv && \
#     cp -r /opt/conda/envs/venv /app/venv

#     RUN ls -l /app/venv 
##### DEPENDENCIES
# deps : 노드 종속성을 설치한다.
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
# Use npm ci if package-lock.json exists, otherwise use npm install
RUN \
    if [ -f package-lock.json ]; then npm ci; \
    else echo "Lockfile not found." && exit 1; \
    fi

RUN ls -l /app/node_modules
    ##### BUILDER
# builder : deps 에서 Node 모듈 폴더를 복사하고 모든 프로젝트 폴더와 파일을 복사한 후 프로덕션을 위한 애플리케이션을 빌드한다.
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# node_modules 확인
RUN ls -l /app/node_modules

RUN \
    if [ -f package-lock.json ]; then SKIP_ENV_VALIDATION=1 npm run build; \
    else echo "package-lock.json not found." && exit 1; \
    fi


##### RUNNER
FROM node:20-alpine AS server
WORKDIR /app

# Python 종속성과 가상환경 복사
# COPY --from=python_deps /app/venv /app/venv
# Conda 가상 환경 활성화
# RUN echo "source activate venv" > ~/.bashrc

COPY --from=builder /app/data ./data
COPY --from=builder /app/model ./model
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/index.js ./index.js
COPY --from=builder /app/query.js ./query.js
COPY --from=builder /app/resolver.py ./resolver.py
COPY --from=builder /app/recommender.py ./recommender.py
COPY --from=builder /app/node_modules ./node_modules  

# node_modules 확인
RUN ls -l /app/node_modules

# 환경 변수 설정
ENV PATH=/opt/conda/envs/venv/bin:$PATH
ENV NODE_ENV production

# Node.js와 Python 실행 포트 설정
EXPOSE 8080
ENV HOSTNAME="0.0.0.0"

# 서버 시작 명령어 (Node.js 서버)
CMD ["node", "index.js"]
