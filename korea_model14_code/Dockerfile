FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY model_14.py .
COPY permutation_importance_14.py .
COPY model_ablation_resume.py .
COPY permutation_importance_selected.py .

# check_data.py를 사용할 경우 필요합니다.
COPY check_data.py .

CMD ["bash"]
