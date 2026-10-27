FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workbench

RUN pip install uv

COPY pyproject.toml .
COPY common/ common/
COPY api/ api/
COPY db/ db/

RUN python3 -c "import tomllib,sys; f=open('pyproject.toml','rb'); t=tomllib.load(f); f.close(); d=t['project']['dependencies']+t['project']['optional-dependencies']['api']; open('/tmp/ar.txt','w').write('\n'.join(d))" \
    && uv pip install --system -r /tmp/ar.txt

ENV PYTHONPATH=/workbench

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
