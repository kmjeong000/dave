FROM woensugchoi/ubuntu-arm-rdp-base:latest
ARG USER=docker

# ROS-Gazebo arg
ARG BRANCH="rover"
ARG ROS_DISTRO="jazzy"

# Update OS
RUN apt update && apt upgrade -y && apt autoremove -y

# Install ROS-Gazebo framework
ADD https://raw.githubusercontent.com/kmjeong000/dave/$BRANCH/\
extras/ros-jazzy-gz-harmonic-install.sh install.sh
RUN sudo bash install.sh

# Install wave sim dependencies
RUN apt-get update && \
    apt-get install -y libcgal-dev libfftw3-dev \
    && rm -rf /var/lib/apt/lists/

# Prereqs for Ardupilot - Ardurover
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
# hadolint ignore=DL3008
ADD --chown=root:root --chmod=0644 https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list /etc/ros/rosdep/sources.list.d/00-gazebo.list
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" |  tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get -q update && \
    apt-get install -y --no-install-recommends \
    python-is-python3 python3-future python3-wxgtk4.0 python3-pexpect \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgz-sim8-dev rapidjson-dev libopencv-dev libasio-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl \
    # mavproxy setting
    python3-dev python3-opencv python3-pip python3-matplotlib python3-lxml \
    && rm -rf /var/lib/apt/lists/
# Install mavros
RUN apt-get update && \
    apt-get -y install --no-install-recommends ros-jazzy-mavros* \
    && rm -rf /tmp/*
WORKDIR /opt/mavros_ws
RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && \
    bash ./install_geographiclib_datasets.sh

# Download the background image from GitHub raw content URL
# hadolint ignore=DL3047
RUN wget -O /usr/share/backgrounds/custom-background.png -q \
    https://raw.githubusercontent.com/kmjeong000/dave/$BRANCH/\
extras/background.png && \
    mv /usr/share/backgrounds/warty-final-ubuntu.png \
        /usr/share/backgrounds/warty-final-ubuntu.png.bak && \
    mv /usr/share/backgrounds/custom-background.png \
        /usr/share/backgrounds/warty-final-ubuntu.png && \
    cp /usr/share/backgrounds/warty-final-ubuntu.png \
        /usr/share/backgrounds/ubuntu-wallpaper-d.png

# Install Ardupilot - Ardurover .docker/ardurover.dockerfile
USER docker
RUN wget -O /tmp/install.sh https://raw.githubusercontent.com/kmjeong000/dave/$BRANCH/extras/ardurover-ubuntu-install-local.sh
RUN chmod +x /tmp/install.sh && bash /tmp/install.sh

# Set up Dave workspace
ENV DAVE_UNDERLAY=/home/$USER/dave_ws
WORKDIR $DAVE_UNDERLAY/src
RUN wget -O /home/$USER/dave_ws/dave.repos -q https://raw.githubusercontent.com/kmjeong000/dave/$BRANCH/\
extras/repos/dave.$ROS_DISTRO.repos
RUN vcs import --shallow --input "/home/$USER/dave_ws/dave.repos"

USER root
# hadolint ignore=DL3027
RUN apt update && apt --fix-broken install && \
    rosdep init && rosdep update --rosdistro $ROS_DISTRO && \
    rosdep install --rosdistro $ROS_DISTRO -iy --from-paths . && \
    rm -rf /var/lib/apt/lists/
USER docker

# Build dave workspace
WORKDIR $DAVE_UNDERLAY
RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" && colcon build

#Build wave sim
WORKDIR $DAVE_UNDERLAY/src/dave/gazebo/dave_gz_world_plugins/ocean-waves
RUN colcon build

#Build wave sim GUI plugin
WORKDIR $DAVE_UNDERLAY/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/src/gui/plugins/waves_control
RUN mkdir build && cd build && cmake .. && make

#Patch for wave sim
USER root
RUN ln -s /opt/ros/jazzy/opt/gz_ogre_next_vendor/lib/libOgreNextMain.so.2.3.3 /opt/ros/jazzy/opt/gz_ogre_next_vendor/lib/libOgreNextMain.so.2.3.1

# Set User as user
USER docker
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /opt/gazebo/install/setup.bash" >> ~/.bashrc && \
    echo "source /opt/mavros/install/setup.bash" >> ~/.bashrc && \
    echo "source $DAVE_UNDERLAY/install/setup.bash" >> ~/.bashrc && \
    echo "export GEOGRAPHICLIB_GEOID_PATH=/usr/local/share/GeographicLib/geoids" >> ~/.bashrc && \
    echo "export PYTHONPATH=\$PYTHONPATH:/opt/gazebo/install/lib/python" >> ~/.bashrc && \
    echo "export PATH=/home/$USER/ardupilot_ws/ardupilot/Tools/autotest:\$PATH" >> ~/.bashrc && \
    echo "export PATH=/home/$USER/ardupilot_ws/ardupilot/build/sitl/bin:\$PATH" >> ~/.bashrc && \
    echo "export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/$USER/ardupilot_ws/ardupilot_gazebo/build:\$GZ_SIM_SYSTEM_PLUGIN_PATH:/home/docker/dave_ws/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib" >> ~/.bashrc && \
    echo "export GZ_GUI_PLUGIN_PATH=\$GZ_GUI_PLUGIN_PATH:/home/docker/dave_ws/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/src/gui/plugins/waves_control/build" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/home/docker/dave_ws/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=/home/$USER/ardupilot_ws/ardupilot_gazebo/models:/home/$USER/ardupilot_ws/ardupilot_gazebo/worlds:\$GZ_SIM_RESOURCE_PATH" >> ~/.bashrc && \
    echo "\n\n" >> ~/.bashrc && echo "if [ -d ~/HOST ]; then chown $USER:$USER ~/HOST; fi" >> ~/.bashrc  && \
    echo "export PS1='\[\e[1;36m\]\u@DAVE_docker\[\e[0m\]\[\e[1;34m\](\$(hostname | cut -c1-12))\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '" >>  ~/.bashrc

# Other environment variables
RUN echo "export XDG_RUNTIME_DIR=~/.xdg_log" >> ~/.bashrc && \
    echo "unset SESSION_MANAGER" >> ~/.bashrc

# Create and activate Python virtual environment
RUN python3 -m venv /home/docker/.venv && \
    . /home/docker/.venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install PyYAML pygame mavproxy pexpect packaging urllib3 empy==3.3.4 future && \
    echo "alias venv='source /home/docker/.venv/bin/activate'" >> ~/.bashrc
ENV PATH="/home/docker/.venv/bin:$PATH"

RUN mkdir -p /usr/local/share/GeographicLib/geoids && \
    ln -s /usr/share/GeographicLib/geoids/egm96-5.pgm /usr/local/share/GeographicLib/geoids/egm96-5.pgm && \
    chmod 644 /usr/share/GeographicLib/geoids/egm96-5.pgm

# Create and write the welcome message to a new file
RUN mkdir -p /home/docker/.config/autostart && \
    printf '\033[1;36m =====\n' >> ~/.hi && \
    printf '  ____    ___     _______      _                     _   _      \n' >> ~/.hi && \
    printf ' |  _ \  / \ \   / | ____|    / \   __ _ _   _  __ _| |_(_) ___ \n' >> ~/.hi && \
    printf ' | | | |/ _ \ \ / /|  _|     / _ \ / _` | | | |/ _` | __| |/ __|\n' >> ~/.hi && \
    printf ' | |_| / ___ \ V / | |___   / ___ | (_| | |_| | (_| | |_| | (__ \n' >> ~/.hi && \
    printf ' |____/_/   \_\_/  |_____| /_/   \_\__, |\__,_|\__,_|\__|_|\___|\n' >> ~/.hi && \
    printf ' __     ___      _               _     _____            _       \n' >> ~/.hi && \
    printf ' \ \   / (_)_ __| |_ _   _  __ _| |   | ____|_ ____   _(_)_ __  \n' >> ~/.hi && \
    printf '  \ \ / /| | `__| __| | | |/ _` | |   |  _| | `_ \ \ / | | `__| \n' >> ~/.hi && \
    printf '   \ V / | | |  | |_| |_| | (_| | |   | |___| | | \ V /| | |_   \n' >> ~/.hi && \
    printf '    \_/  |_|_|   \__|\__,_|\__,_|_|   |_____|_| |_|\_/ |_|_(_)  \n\033[0m' >> ~/.hi && \
    printf '\033[1;32m\n =====\n\033[0m' >> ~/.hi && \
    printf "\\033[1;32m 👋 Hi! This is Docker virtual environment for DAVE\n\\033[0m" \
    >> ~/.hi && \
    printf "\\033[1;33m\tROS2 Jazzy - Gazebo Harmonic (w ardupilot(ardurover) + mavros)\n\n\n\\033[0m" \
    >> ~/.hi && \
    printf "\\033[1;33m\t💡 Virtual environment shortcut: 'venv' (type this command to activate the environment)\n\n\n\\033[0m" \
    >> ~/.hi

# Remove sudo message
RUN touch /home/docker/.sudo_as_admin_successful

# Autostart terminal
# hadolint ignore=SC3037
RUN echo "[Desktop Entry]\nType=Application" \
    > /home/docker/.config/autostart/terminal.desktop && \
    echo "Exec=gnome-terminal -- bash -c 'cat ~/.hi; exec bash'" \
    >> /home/docker/.config/autostart/terminal.desktop && \
    echo -e "X-GNOME-Autostart-enabled=true" \
    >> /home/docker/.config/autostart/terminal.desktop