ARG ROS_DISTRO="jazzy"
FROM osrf/ros:${ROS_DISTRO}-desktop-full
ARG ROS_DISTRO
ARG BRANCH="rover"

ARG USERNAME=docker

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN useradd -m -s /bin/bash ${USERNAME} 2>/dev/null || true && \
    mkdir -p /tmp/runtime-${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /tmp/runtime-${USERNAME}

# Install Utilities
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo xterm init systemd snapd vim net-tools \
    curl wget git build-essential cmake cppcheck \
    gnupg libeigen3-dev libgles2-mesa-dev \
    lsb-release pkg-config protobuf-compiler \
    python3-dbg python3-pip python3-venv python3-pexpect \
    python-is-python3 python3-future python3-wxgtk4.0 \
    qtbase5-dev ruby dirmngr gnupg2 nano xauth \
    software-properties-common htop libtool \
    x11-apps mesa-utils bison flex automake \
    locales tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Locale for UTF-8
RUN truncate -s0 /tmp/preseed.cfg && \
   (echo "tzdata tzdata/Areas select Etc" >> /tmp/preseed.cfg) && \
   (echo "tzdata tzdata/Zones/Etc select UTC" >> /tmp/preseed.cfg) && \
   debconf-set-selections /tmp/preseed.cfg && \
   rm -f /etc/timezone && \
   dpkg-reconfigure -f noninteractive tzdata
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get -y install --no-install-recommends locales tzdata \
    && rm -rf /tmp/*
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install ROS-Gazebo framework
ADD https://raw.githubusercontent.com/kmjeong000/dave/${BRANCH}/extras/ros-jazzy-gz-harmonic-install.sh /tmp/install.sh
RUN bash /tmp/install.sh && rm -f /tmp/install.sh

# Install wave sim dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcgal-dev libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Prereqs for ArduPilot / ArduRover
RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    wget -O /etc/ros/rosdep/sources.list.d/00-gazebo.list \
    https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list && \
    chown root:root /etc/ros/rosdep/sources.list.d/00-gazebo.list && \
    chmod 0644 /etc/ros/rosdep/sources.list.d/00-gazebo.list

RUN wget https://packages.osrfoundation.org/gazebo.gpg \
        -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get -q update && \
    apt-get install -y --no-install-recommends \
    python-is-python3 python3-future python3-wxgtk4.0 python3-pexpect \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgz-sim8-dev rapidjson-dev libopencv-dev libasio-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl \
    python3-dev python3-opencv python3-pip python3-matplotlib python3-lxml \
    && rm -rf /var/lib/apt/lists/*

# Install MAVROS
RUN apt-get update && \
    apt-get -y install --no-install-recommends ros-${ROS_DISTRO}-mavros\* \
    && rm -rf /var/lib/apt/lists/*

# Install GeographicLib datasets
WORKDIR /opt/mavros_ws
RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && \
    bash ./install_geographiclib_datasets.sh && \
    rm -f ./install_geographiclib_datasets.sh

# Install ArduRover
RUN wget -O /tmp/install_ardurover.sh \
    https://raw.githubusercontent.com/kmjeong000/dave/${BRANCH}/extras/ardurover-ubuntu-install-local.sh && \
    chmod +x /tmp/install_ardurover.sh && \
    bash /tmp/install_ardurover.sh && \
    cd /home/${USERNAME}/ardupilot_ws/ardupilot && \
    ./waf clean && \
    ./waf configure --board sitl && \
    ./waf --targets bin/ardurover && \
    test -f /home/${USERNAME}/ardupilot_ws/ardupilot/build/sitl/bin/ardurover && \
    rm -f /tmp/install_ardurover.sh

# Install QGroundControl
RUN mkdir -p /opt/QGC && cd /opt/QGC && \
    wget -O /opt/QGC/QGroundControl-x86_64.AppImage \
    "https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-x86_64.AppImage" && \
    chmod +x QGroundControl-x86_64.AppImage && \
    ./QGroundControl-x86_64.AppImage --appimage-extract && \
    mv squashfs-root/* /opt/QGC/ && \
    rm -rf squashfs-root QGroundControl-x86_64.AppImage && \
    ln -sf /opt/QGC/AppRun /usr/local/bin/qgroundcontrol

# Install Firefox
RUN curl -L "https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64&lang=en-US" \
        -o /tmp/firefox.tar.xz && \
    mkdir -p /opt/firefox && \
    tar -xJf /tmp/firefox.tar.xz -C /opt && \
    ln -sf /opt/firefox/firefox /usr/local/bin/firefox && \
    rm -f /tmp/firefox.tar.xz

# Set up Dave workspace
ENV DAVE_WS=/opt/dave_ws
WORKDIR $DAVE_WS/src

ADD https://raw.githubusercontent.com/kmjeong000/dave/${BRANCH}/extras/repos/dave.${ROS_DISTRO}.repos $DAVE_WS/dave.repos
RUN vcs import --shallow --input $DAVE_WS/dave.repos

# Install dave dependencies
RUN rosdep init 2>/dev/null || true && \
    apt-get update && \
    apt-get --fix-broken install -y && \
    rosdep update --rosdistro ${ROS_DISTRO} && \
    rosdep install --rosdistro ${ROS_DISTRO} -iy --from-paths . && \
    rm -rf /var/lib/apt/lists/*

# Compile Dave
WORKDIR $DAVE_WS
RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" && \
    colcon build --symlink-install

# Build wave sim
WORKDIR $DAVE_WS/src/dave/gazebo/dave_gz_world_plugins/ocean-waves
RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" && \
    colcon build

# Build wave sim GUI plugin
WORKDIR $DAVE_WS/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/src/gui/plugins/waves_control
RUN mkdir -p build && cd build && cmake .. && make

# Patch for wave sim
RUN if [ -f /opt/ros/${ROS_DISTRO}/opt/gz_ogre_next_vendor/lib/libOgreNextMain.so.2.3.3 ]; then \
        ln -sf /opt/ros/${ROS_DISTRO}/opt/gz_ogre_next_vendor/lib/libOgreNextMain.so.2.3.3 \
               /opt/ros/${ROS_DISTRO}/opt/gz_ogre_next_vendor/lib/libOgreNextMain.so.2.3.1; \
    fi

ENV PATH="/home/${USERNAME}/ardupilot_ws/ardupilot/Tools/autotest:/home/${USERNAME}/ardupilot_ws/ardupilot/build/sitl/bin:${PATH}"

# GeographicLib compatibility link
RUN mkdir -p /usr/local/share/GeographicLib/geoids && \
    ln -sf /usr/share/GeographicLib/geoids/egm96-5.pgm /usr/local/share/GeographicLib/geoids/egm96-5.pgm && \
    chmod 644 /usr/share/GeographicLib/geoids/egm96-5.pgm

WORKDIR /root
CMD ["/bin/bash"]