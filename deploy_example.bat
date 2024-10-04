@echo off
REM .env 파일에서 환경 변수 읽기
for /f "tokens=1,2 delims==" %%A in (".env") do (
    if "%%A"=="DOCKER_USERNAME" set DOCKER_USERNAME=%%B
    if "%%A"=="DOCKER_PASSWORD" set DOCKER_PASSWORD=%%B
)

REM Docker 이미지 이름과 태그 설정
set IMAGE_NAME=your-dockerhub-username/your-image-name
set TAG=latest

REM Docker 빌드 명령어 실행
echo Building Docker image...
docker build -t %IMAGE_NAME%:%TAG% .

REM Docker Hub 로그인
echo Logging in to Docker Hub...
docker login -u %DOCKER_USERNAME% -p %DOCKER_PASSWORD%

REM Docker 이미지 푸시
echo Pushing Docker image to Docker Hub...
docker push %IMAGE_NAME%:%TAG%

REM 커밋 메시지 확인
if "%1"=="" (
    echo Error: Commit message is required.
    exit /b 1
)

REM Git add, commit, push 명령어 실행
echo Staging all changes...
git add .

echo Committing changes with message: %1
git commit -m "%1"

echo Pushing changes to remote repository...
git push

echo Git commit, push, and Docker build/push complete!
pause
