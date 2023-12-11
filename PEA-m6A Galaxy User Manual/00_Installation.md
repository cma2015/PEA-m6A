---
typora-root-url: ./img
---

# <center>PEA-m6A User Manual (version 1.0)</center>

- PEA-m6A is an ensemble learning framework for predicting m6A modifications at regional-scale.
- PEA-m6A consists of four modules:**Sample Preparation, Feature Encoding, Model Development and Model Assessment**, each of which contains a comprehensive collection of functions with pre-specified parameters available.
- PEA-m6A was powered with an advanced  packaging technology, which enables compatibility and portability.
- PEA-m6A project is hosted on http://github.com/cma2015/PEA-m6A
- PEA-m6A docker image is available at http://hub.docker.com/r/malab/peam6a
- PEA-m6A server can be accessed via http://peam6a.omstudio.cloud

## PEA-m6A installation

- **Step 1**: Docker installation

  **i) Docker installation and start ([Official installation tutorial](https://docs.docker.com/install))**

  For **Windows (Only available for Windows 10 Prefessional and Enterprise version):**

  - Download [Docker](https://download.docker.com/win/stable/Docker for Windows Installer.exe) for windows;
  - Double click the EXE file to open it;
  - Follow the wizard instruction and complete installation;
  - Search docker, select **Docker for Windows** in the search results and click it.

  For **Mac OS X (Test on macOS Sierra version 10.12.6 and macOS High Sierra version 10.13.3):**

  - Download [Docker](https://download.docker.com/mac/stable/Docker.dmg) for Mac OS;

  - Double click the DMG file to open it;
  - Drag the docker into Applications and complete installation;
  - Start docker from Launchpad by click it.

  For **Ubuntu (Test on Ubuntu 18.04 LTS):**

  - Go to [Docker](https://download.docker.com/linux/ubuntu/dists/), choose your Ubuntu version, browse to **pool/stable** and choose **amd64, armhf, ppc64el or s390x**. Download the **DEB** file for the Docker version you want to install;

  - Install Docker, supposing that the DEB file is download into following path:**"/home/docker-ce~ubuntu_amd64.deb"**

    ```
      $ sudo dpkg -i /home/docker-ce<version-XXX>~ubuntu_amd64.deb      
      $ sudo apt-get install -f
    ```

​	**ii) Verify if Docker is installed correctly**

​	Once Docker installation is completed, we can run `hello-world` image to verify if Docker is installed correctly. Open terminal in Mac OS X and Linux operating system and open CMD for Windows operating system, then type the following command:

```
 $ docker run hello-world
```

**Note:** root permission is required for Linux operating system.

- **Step 2**: PEA-m6A installation from Docker Hub

```
# pull latest peam6a Docker image from docker hub
$ docker pull malab/peam6a
```

- **Step 3**: Launch PEA-m6A local server

```
$ docker run -it -p 8090:8090 malab/peam6a bash
$ bash /home/galaxy/run.sh
```

Then, PEA-m6A local server can be accessed via [http://localhost:8090](http://localhost:8090/)

![home](/home.png)

