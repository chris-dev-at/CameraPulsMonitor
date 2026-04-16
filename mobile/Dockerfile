FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    openjdk-17-jdk \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

RUN mkdir -p $ANDROID_HOME && \
    cd /tmp && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O cmdline-tools.zip && \
    unzip -q cmdline-tools.zip && \
    mkdir -p $ANDROID_HOME/cmdline-tools && \
    mv cmdline-tools $ANDROID_HOME/cmdline-tools/latest && \
    rm cmdline-tools.zip

RUN yes | $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --licenses > /dev/null 2>&1 || true

RUN $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \
    "platform-tools" \
    "platforms;android-34" \
    "build-tools;34.0.0"

RUN wget -q https://services.gradle.org/distributions/gradle-8.4-bin.zip -O /tmp/gradle.zip && \
    unzip -q /tmp/gradle.zip -d /opt && \
    rm /tmp/gradle.zip

ENV GRADLE_HOME=/opt/gradle-8.4
ENV PATH=$PATH:$GRADLE_HOME/bin

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
COPY capacitor.config.json ./
COPY www/ ./www/
COPY assets/ ./assets/
COPY android ./android

RUN npm install @capacitor/cli @capacitor/core && \
    npx cap sync android

RUN cd android && chmod +x gradlew && ./gradlew assembleDebug --no-daemon -q

RUN mkdir -p /output && cp android/app/build/outputs/apk/debug/*.apk /output/