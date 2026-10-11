# 后端容器从声明依赖构建，不读取宿主环境密钥。
FROM python:3.12-slim
WORKDIR /workspace
COPY backend /workspace/backend
COPY knowledge/v1.2 /workspace/knowledge/v1.2
COPY scripts /workspace/scripts
RUN pip install --no-cache-dir ./backend
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
