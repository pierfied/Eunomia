FROM continuumio/anaconda3:latest

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install cmake, gcc, and g++
RUN apt-get update && \
    apt-get install -y cmake g++ && \
    apt-get clean

# Install Chronos
RUN git clone https://github.com/pierfied/Chronos.git && \
    cd Chronos && \
    cmake .; make && \
    cp src/*.h /usr/local/include/ && \
    cp libchronos.so /usr/local/lib/ && \
    cd .. && \
    rm -rf Chronos/
