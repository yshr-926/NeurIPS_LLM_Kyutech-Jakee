FROM pytorch/pytorch:latest

ENV TZ JST-9
ENV TERM xterm
ENV NVIDIA_VISIBLE_DEVICES all 
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update                                                     && \
    apt-get -y install                                                    \
      locales                                                             \ 
      git                                                              && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8                            && \
    pip install --no-cache-dir                                            \
                --upgrade                                                 \
                  pip                                                     \
                  setuptools                                           && \
    pip install --no-cache-dir                                            \
                  tqdm                                                 && \
    pip install --no-cache-dir                                            \
                  matplotlib                                           && \
    pip install --no-cache-dir                                            \
                  pandas                                               && \
    apt-get clean                                                      && \
    rm -rf /var/lib/apt/lists/*             
