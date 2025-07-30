FROM python:3.9-slim

WORKDIR /home/app

COPY requirements.txt .
COPY test_cuda.py .

# If a GPU is available:
RUN pip install --upgrade pip && \
    pip install git+https://github.com/facebookresearch/segment-anything.git && \
    pip install --no-cache-dir -r requirements.txt
RUN conda install cudatoolkit=11.3 -c pytorch
RUN python test_cuda.py

# Otherwise:
# RUN pip install --upgrade pip && \
#     pip install git+https://github.com/facebookresearch/segment-anything.git && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install cpuonly -c pytorch

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--volume", ".:."]