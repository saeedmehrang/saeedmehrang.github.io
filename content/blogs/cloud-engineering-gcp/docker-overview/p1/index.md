---
title: "Docker Fundamentals: Containers vs VMs (part 1 of 6)"
date: 2025-10-31
draft: false
author: Saeed Mehrang
description: "Learn Docker basics, understand how containers differ from virtual machines, explore Docker architecture, and run your first container in this beginner-friendly guide."
summary: "A comprehensive introduction to Docker covering containerization concepts, Docker architecture components, the difference between containers and VMs, and hands-on commands to get started with your first container."
tags: ["docker", "containers", "devops", "cloud-engineering", "virtualization"]
categories: ["Docker Tutorial Series"]
series: ["Docker & Docker Compose Miniseries"]
weight: 1
ShowToc: true
TocOpen: true
math: true
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 10-12 minutes |
| **Technical Level** | Beginner |
| **Prerequisites** | Basic command-line knowledge |

---

This is the first part out of 6 in this miniseries. See the table below for the topics that are covered.


| # | Blog Title | Key Topics Covered |
|---|------------|-------------------|
| **1** | **Docker Fundamentals: Containers vs VMs** | **What is Docker • Containers vs VMs • Docker architecture • Images, Containers, Registries • Installation • Basic commands (`run`, `ps`, `images`, `stop`, `rm`) • Container lifecycle** |
| 2 | Building Your First Docker Image | Dockerfile basics • Instructions (FROM, RUN, COPY, CMD, ENTRYPOINT, EXPOSE) • Building images • Image layers & caching • Tagging • Multi-stage builds intro • .dockerignore • Docker Hub |
| 3 | Docker Networking & Storage | Network fundamentals • Network types (bridge, host, none, overlay) • Port mapping • Container communication • Volumes vs bind mounts • Data persistence • Volume management |
| 4 | Docker Compose Essentials | Why Compose • docker-compose.yml structure • Services, networks, volumes • Compose commands (`up`, `down`, `ps`, `logs`, `exec`) • Multi-container apps • Container naming & discovery |
| 5 | Advanced Compose: Dependencies & Environment Management | Service dependencies • Health checks • Environment variables • .env files • Docker secrets • Compose profiles • Scaling services • Configuration overrides |
| 6 | Docker Best Practices & Optimization | Image optimization • Security best practices • Multi-stage builds deep dive • Logging strategies • Resource limits • Restart policies • Monitoring basics • Common pitfalls • Next steps |


## Introduction: The Containerization Revolution

Remember the days when deploying an application meant wrestling with dependencies, library versions, and the dreaded "it works on my machine" problem? Those frustrations led to one of the most transformative technologies in modern software development: containerization.

Docker has revolutionized how we build, ship, and run applications. Since its release in 2013, it has become the industry standard for containerization, with millions of developers worldwide using it daily. According to recent surveys, Docker is one of the most popular development tools, with over 50% of developers relying on it for their workflows.

In this first installment of our Docker miniseries, we'll explore what makes Docker special, how it differs from traditional virtualization, and get you running your first container.

## What is Docker?

Docker is an open-source platform that enables developers to package applications and their dependencies into lightweight, portable units called **containers**. Think of a container as a standardized package that includes everything your application needs to run: code, runtime, system tools, libraries, and settings.

The key advantage? A containerized application will run consistently across any environment—whether on your laptop, a colleague's machine, a testing server, or in production. This solves the age-old problem of environmental inconsistencies that have plagued software development for decades.

Docker uses a client-server architecture where the Docker client communicates with the Docker daemon, which handles the heavy lifting of building, running, and managing containers. This architecture makes Docker both powerful and easy to use.

## Containers vs Virtual Machines

To understand why containers are revolutionary, let's compare them to virtual machines (VMs)—a technology you might already be familiar with.

### Virtual Machines

Virtual machines abstract the entire hardware server. Each VM runs a complete operating system on top of a hypervisor, which sits on the host OS. Here's what this looks like:

```
┌─────────────────────────────────────────┐
│         Application Layer               │
├─────────────────────────────────────────┤
│         Guest OS (Full OS)              │
├─────────────────────────────────────────┤
│         Hypervisor                      │
├─────────────────────────────────────────┤
│         Host OS                         │
├─────────────────────────────────────────┤
│         Physical Hardware               │
└─────────────────────────────────────────┘
```

Each VM includes:
- A full operating system (several gigabytes)
- All OS libraries and binaries
- Significant overhead in terms of resources
- Longer startup times (minutes)

### Docker Containers

Containers take a different approach. Instead of abstracting the hardware, they abstract the operating system kernel. Multiple containers share the same OS kernel but run in isolation from each other.

```
┌─────────────────────────────────────────┐
│    App A    │    App B    │    App C    │
├─────────────┼─────────────┼─────────────┤
│   Libs/Bins │  Libs/Bins  │  Libs/Bins  │
├─────────────────────────────────────────┤
│         Docker Engine                   │
├─────────────────────────────────────────┤
│         Host OS / **Shared Kernel**     │
├─────────────────────────────────────────┤
│         Physical Hardware               │
└─────────────────────────────────────────┘
```

Key differences:

| Aspect | Virtual Machines | Docker Containers |
|--------|------------------|-------------------|
| **Size** | GBs (full OS) | MBs (app + dependencies only) |
| **Startup Time** | Minutes | Seconds |
| **Resource Usage** | Heavy (dedicated resources per VM) | Lightweight (shares host kernel) |
| **Isolation** | Complete OS-level isolation | Process-level isolation |
| **Portability** | Less portable (hypervisor dependent) | Highly portable (runs anywhere Docker runs) |
| **Performance** | Slower (additional OS overhead) | Near-native performance |

Containers are not VM replacements—they serve different purposes. VMs provide complete isolation and can run different operating systems, while containers excel at application portability, resource efficiency, and rapid deployment.

## Docker Architecture: Docker Engine, Daemon, and Client

Docker uses a client-server architecture with three main components working together. Let's break down each one.

### Docker Engine

Docker Engine is the core of Docker. It's a client-server application consisting of:

1. **Server (Docker Daemon)** - A long-running background process called `dockerd`
2. **REST API** - Interfaces that programs use to communicate with the daemon
3. **Client (Docker CLI)** - The command-line interface you interact with

### Docker Daemon (dockerd)

The Docker daemon is the brain of the operation. It's a persistent background process running on your host machine that:

- Listens for Docker API requests from clients
- Manages all Docker objects: images, containers, networks, and volumes
- Handles building, running, and distributing containers
- Can communicate with other daemons for distributed setups
- Uses components like containerd and runc to actually run containers

The daemon does the heavy lifting. When you issue a command, the daemon processes it and takes the necessary actions.

### Docker Client (CLI)

The Docker client is your primary interface. When you type commands like `docker run` or `docker build`, you're using the client. The client:

- Translates your commands into REST API requests
- Sends these requests to the Docker daemon
- Can communicate with local or remote daemons
- Provides command-line access to all Docker functionality

Here's how they interact:

```
You → docker run nginx → Docker Client → REST API → Docker Daemon → Container Created
```

### Docker Desktop

For Windows, macOS, and Linux desktop users, Docker Desktop provides a graphical interface that includes the Docker Engine, client, CLI tools, and a dashboard for managing containers, images, and volumes. It makes getting started with Docker much easier on desktop operating systems.

## Key Concepts: Images, Containers, and Registries

Before running commands, let's understand three fundamental concepts.

### Docker Images

A **Docker image** is a read-only template that contains instructions for creating a container. Think of it as a recipe or blueprint. An image includes:

- Application code
- Runtime environment (like Node.js or Python)
- System libraries and tools
- Environment variables
- Configuration files

Images are built in layers. Each instruction in a Dockerfile creates a new layer. These layers are cached and reused, making builds faster and more efficient.

### Docker Containers

A **container** is a running instance of an image. If an image is a recipe, a container is the actual dish you've prepared. A container is a running instance of an image. It adds a thin, writable layer on top of the read-only image layers, which is where all runtime changes occur. Containers:

- Are isolated processes on your host system
- Have their own filesystem, network, and process space
- Can be started, stopped, moved, and deleted
- Share the host OS kernel but run in isolation
- Are lightweight and start in seconds

Multiple containers can run from the same image, each operating independently.

### Docker Registries

A **Docker registry** stores and distributes Docker images. The most popular registry is Docker Hub, which hosts millions of images including:

- **Official images** from companies like Ubuntu, Redis, and Nginx
- **Community images** created by developers worldwide
- **Private images** for proprietary applications

When you run a container, Docker first checks if the image exists locally. If not, it pulls the image from the configured registry (Docker Hub by default).

## Installing Docker

Let's get Docker installed on your system. Docker supports Linux, macOS, and Windows.

### Linux (Ubuntu/Debian)

The recommended way to install Docker on Linux is using Docker's official repository is below. See the installation guide for all linux kernels here https://docs.docker.com/engine/install/.

```bash
# Update package index
sudo apt update

# Install prerequisites
sudo apt install -y ca-certificates curl gnupg

# Add Docker's GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
docker --version
```

Alternatively, use the convenience script (with caution, check it first before run it!):

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### macOS and Windows

For desktop operating systems, download and install Docker Desktop:

1. Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Download the installer for your OS
3. Run the installer and follow the prompts
4. Launch Docker Desktop
5. Verify installation by opening a terminal and running: `docker --version`

**Requirements:**
- **macOS**: Version 10.15 or higher, minimum 4GB RAM
- **Windows**: Windows 10/11 Pro, Enterprise, or Education with WSL 2 enabled

### Verifying Your Installation

After installation, verify Docker is working:

```bash
docker --version
# Output: Docker version 27.5.1, build...

docker run hello-world
```

The `hello-world` container will download a test image and run it. If you see a welcome message, Docker is correctly installed.

## First Commands: Getting Started with Docker

Now that Docker is installed, let's explore the essential commands you'll use daily.

### docker run - Starting Containers

The `docker run` command creates and starts a new container:

```bash
docker run [OPTIONS] IMAGE [COMMAND]
```

When you run this command, Docker:
1. Checks if the image exists locally
2. Pulls the image from Docker Hub if not found
3. Creates a new container from the image
4. Starts the container
5. Executes any specified commands

### docker ps - Viewing Running Containers

To see all running containers:

```bash
docker ps
```

To see all containers, including stopped ones:

```bash
docker ps -a
```

Output includes container ID, image, command, creation time, status, and names.

### docker images - Listing Images

View all images stored locally:

```bash
docker images
```

This shows the repository, tag, image ID, creation date, and size for each image.

### docker stop - Stopping Containers

To stop a running container:

```bash
docker stop CONTAINER_ID
# or
docker stop CONTAINER_NAME
```

The container stops gracefully. You can use the first few characters of the container ID—Docker is smart enough to figure it out.

### docker rm - Removing Containers

To delete a stopped container:

```bash
docker rm CONTAINER_ID
```

To remove a running container, add the `-f` flag to force removal:

```bash
docker rm -f CONTAINER_ID
```

## Code Demo: Running a Simple Nginx Container

Let's put everything together by running a real web server using Nginx.

### Step 1: Pull and Run Nginx

```bash
docker run -d -p 8080:80 --name my-nginx nginx
```

Let's break down this command:
- `docker run` - Creates and starts a container
- `-d` - Runs in detached mode (background)
- `-p 8080:80` - Maps port 8080 on your host to port 80 in the container
- `--name my-nginx` - Gives the container a friendly name
- `nginx` - The image to use

### Step 2: Verify It's Running

```bash
docker ps
```

You should see your nginx container listed with status "Up".

### Step 3: Access Nginx

Open your browser and navigate to `http://localhost:8080`. You should see the Nginx welcome page.

### Step 4: View Container Logs

```bash
docker logs my-nginx
```

This shows Nginx access logs and any errors.

### Step 5: Execute Commands Inside the Container

You can run commands inside a running container with the command below. 

```bash
docker exec -it my-nginx bash
```

Once you run this in your terminal you should see your cursor changing to something like this `root@b927dae64901:/#` which is a sign for an interactive shell bash. `root` is the traditional name for the superuser account in Unix-like operating systems, including Linux, with all privileges. Try running:

```bash
ls
cat /etc/nginx/nginx.conf
exit
```

### Step 6: Stop and Remove the Container

```bash
docker stop my-nginx
docker rm my-nginx
```

Alternatively, remove it in one command:

```bash
docker rm -f my-nginx
```

## Understanding Container Lifecycle

Containers go through several states during their lifetime:

1. **Created** - Container is created but not started
2. **Running** - Container is actively running
3. **Paused** - Container processes are paused
4. **Stopped** - Container has exited (can be restarted)
5. **Removed** - Container is deleted

Key lifecycle commands:

```bash
docker create IMAGE      # Create but don't start
docker start CONTAINER   # Start a stopped container
docker restart CONTAINER # Restart a container
docker pause CONTAINER   # Pause a running container
docker unpause CONTAINER # Unpause a paused container
docker stop CONTAINER    # Stop a running container
docker kill CONTAINER    # Force stop immediately
docker rm CONTAINER      # Remove a stopped container
```

When you run `docker run`, it combines `docker create` and `docker start` into one command.

## Key Takeaways

Let's recap what we've covered:

1. **Docker revolutionizes deployment** by packaging applications with all their dependencies into portable containers that run consistently across any environment.

2. **Containers differ from VMs** in that they share the host OS kernel, making them lightweight (MBs vs GBs), fast to start (seconds vs minutes), and more resource-efficient.

3. **Docker architecture** uses a client-server model where the Docker Client sends commands to the Docker Daemon via a REST API. The daemon manages all container operations.

4. **Three core concepts** form Docker's foundation: Images (blueprints), Containers (running instances), and Registries (storage for images).

5. **Essential commands** you'll use constantly include `docker run` (start containers), `docker ps` (view containers), `docker images` (list images), `docker stop` (stop containers), and `docker rm` (remove containers).

6. **Containers have lifecycles** moving through created, running, paused, stopped, and removed states. Understanding this lifecycle helps you manage containers effectively.

In the next blog, we'll dive deeper into creating your own Docker images using Dockerfiles, exploring image layers, and learning best practices for building efficient, production-ready images.

---

**Series Navigation**:
- **Part 1: Docker Fundamentals** (You are here)
- Part 2: Building Your First Docker Image
- Part 3: Docker Networking & Storage
- Part 4: Docker Compose Essentials
- Part 5: Advanced Compose: Dependencies & Environment Management
- Part 6: Docker Best Practices & Optimization


**Next in Series**: [Building Your First Docker Image](../p2) - Learn to create custom Docker images with Dockerfiles, understand layering, and implement multi-stage builds.
