FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && \
	apt-get install -y \
		libsm6 libxrender1 libfontconfig1 \
		git vim && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt



# COPY . /app


# Add locally downloaded models
COPY models /app/V3/server

# RUN chown -R 999 /app && \
# 	chmod -R 777 /app

# CMD cd /app/V3/server && python server.py -cpu

