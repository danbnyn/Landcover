# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Set the working directory in the container
WORKDIR /app

# Copy the my_dataset folder into the container
COPY my_dataset /app/my_dataset

# Install TensorFlow and TensorFlow Datasets
RUN pip install tensorflow tensorflow-datasets numpy pathlib tqdm array-record dm_pix tifffile


# Mount the data and output directories
VOLUME ["/data", "/output"]

# Set the entrypoint to run the tfds build command from within the my_dataset directory
ENTRYPOINT ["tfds", "build", "my_dataset" ,"--file_format", "array_record", "--data_dir", "/output"]
