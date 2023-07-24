# Use an official CUDA runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip

# Set the working directory in the container to /research_capstone
WORKDIR /research_capstone

# Add the current directory contents into the container at /research_capstone
ADD . /research_capstone

# Upgrade pip and Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Jupyter notebook
RUN pip install jupyterlab

# Make port 8888 available to the world outside this container (for Jupyter)
EXPOSE 8888

# Define environment variable
ENV NAME research_capstone

# Run Jupyter notebook when the container launches
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
