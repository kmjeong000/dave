FROM rover:local

ARG USERNAME=docker
USER root

COPY .docker/patches/sailboat_debug.patch /tmp/sailboat_debug.patch
WORKDIR /home/${USERNAME}/ardupilot_ws/ardupilot

RUN git apply /tmp/sailboat_debug.patch && \
    ./waf configure --board sitl && \
    ./waf --targets bin/ardurover

USER ${USERNAME}
WORKDIR /home/${USERNAME}

ENTRYPOINT ["/usr/local/bin/sailboat_entrypoint.sh"]
CMD []