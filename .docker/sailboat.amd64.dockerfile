FROM dave:rover

ARG ROS_DISTRO="jazzy"
ARG USERNAME=docker

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends iproute2 && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/user/1000 && chown ${USERNAME}:${USERNAME} /run/user/1000

USER root
RUN mkdir -p /home/${USERNAME}/sailboat_ws/src && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/sailboat_ws

USER ${USERNAME}
WORKDIR /home/${USERNAME}

ENV SAILBOAT_VENV=/home/${USERNAME}/venv
RUN python3 -m venv --system-site-packages "${SAILBOAT_VENV}" && \
    "${SAILBOAT_VENV}/bin/pip" install --no-cache-dir --upgrade "pip<26" "setuptools<80" "wheel" && \
    "${SAILBOAT_VENV}/bin/pip" install --no-cache-dir pymavlink pyyaml  MAVProxy

ENV SAILBOAT_WS=/home/${USERNAME}/sailboat_ws

ARG SAILBOAT_CACHE_BUST=0
RUN echo "SAILBOAT_CACHE_BUST=$SAILBOAT_CACHE_BUST" >/dev/null
COPY --chown=$USERNAME:$USERNAME . $SAILBOAT_WS/src/dave/
WORKDIR $SAILBOAT_WS
RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" && \
    . "/opt/dave_ws/install/setup.sh" && \
    colcon build

USER root
# COPY --chown=${USERNAME}:${USERNAME} world/gui.config /opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/gui/gui.config
RUN mkdir -p /tmp/runtime-${USERNAME} && \
    chown ${USERNAME}:${USERNAME} /tmp/runtime-${USERNAME} && \
    chmod 700 /tmp/runtime-${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /opt/asv_sim_ws/install/setup.bash" >> ~/.bashrc && \
    echo "if [ -f ${DAVE_WS}/install/setup.bash ]; then source ${DAVE_WS}/install/setup.bash; fi" >> ~/.bashrc && \
    echo "if [ -f ${SAILBOAT_VENV}/bin/activate ]; then source ${SAILBOAT_VENV}/bin/activate; fi" >> ~/.bashrc && \
    echo "if [ -f ${SAILBOAT_WS}/install/setup.bash ]; then source ${SAILBOAT_WS}/install/setup.bash; fi" >> ~/.bashrc && \
    echo "export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/asv_sim_ws/install/lib:/home/${USERNAME}/ardupilot_ws/ardupilot_gazebo/build:\$GZ_SIM_SYSTEM_PLUGIN_PATH:${DAVE_WS}/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=${DAVE_WS}/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/models:/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/worlds:/home/${USERNAME}/ardupilot_ws/ardupilot_gazebo/models:/home/${USERNAME}/ardupilot_ws/ardupilot_gazebo/worlds:\$GZ_SIM_RESOURCE_PATH" >> ~/.bashrc && \
    echo "export XDG_RUNTIME_DIR=/tmp/runtime-${USERNAME}" >> ~/.bashrc && \
    echo "unset SESSION_MANAGER" >> ~/.bashrc && \
    echo "alias venv='source /home/${USERNAME}/venv/bin/activate'" >> ~/.bashrc && \
    echo "alias qgc='unset WAYLAND_DISPLAY; export QT_QPA_PLATFORM=xcb; export XDG_RUNTIME_DIR=/tmp/runtime-${USERNAME}; qgroundcontrol'" >> ~/.bashrc && \
    echo "export PS1='\[\e[1;36m\]\u@DAVE_docker\[\e[0m\]\[\e[1;34m\](\$(hostname | cut -c1-12))\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '" >> ~/.bashrc

COPY .docker/sailboat_entrypoint.sh /usr/local/bin/sailboat_entrypoint.sh
USER root
RUN chmod 0755 /usr/local/bin/sailboat_entrypoint.sh && \
    chmod 0755 /home/docker/sailboat_ws/src/dave/.docker/sailboat_entrypoint.sh
USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENTRYPOINT [ "/usr/local/bin/sailboat_entrypoint.sh" ]
CMD []