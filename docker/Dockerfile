FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
COPY optix7_installer optix7_installer
RUN bash optix7_installer --skip-license --prefix=/root --include-subdir

RUN apt update && apt install -y git \
  clang-9 libc++-9-dev libc++abi-9-dev cmake ninja-build wget libopenexr-dev \
  libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  python3-dev python3-distutils python3-setuptools ffmpeg tmux htop \
  python3-pytest python3-pytest-xdist python3-numpy wget imagemagick \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Mitsuba install
ENV CC=clang-9
ENV CXX=clang++-9
ENV CUDACXX=/usr/local/cuda/bin/nvcc

RUN /opt/conda/bin/python3.7 -m pip install pytest pytest-xdist
RUN git clone --recursive https://github.com/mitsuba-renderer/mitsuba2.git /mitsuba2
ENV PATH=/usr/local/cuda/:/usr/local/cuda/include:${PATH}

WORKDIR /mitsuba2
RUN git pull origin master && git submodule update
COPY mitsuba.conf /mitsuba2/mitsuba.conf
RUN mkdir build \
    && cd build \
    && cmake -GNinja .. \ 
    && ninja

ENV PYTHONPATH=/mitsuba2/dist/python:/mitsuba2/build/dist/python:$PYTHONPATH
ENV PATH=/mitsuba2/dist:/mitsuba2/build/dist:$PATH

RUN echo 'source /mitsuba2/setpath.sh' >> ~/.zshrc
RUN echo 'cd /' >> ~/.zshrc

# Install Zsh to make life easier
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo "export THEME=agnoster" >> ~/.zshrc

RUN pip install cox robustness kornia scikit-image opencv-python imgaug zmq OpenEXR ipdb filelock
RUN wget https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt\?raw=1 -O /cifar_nat.pt 
WORKDIR /
RUN convert -size 200x200 xc:red /red.jpg
RUN wget "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D" -O /robust_imagenet_model.pt

CMD ["zsh"]