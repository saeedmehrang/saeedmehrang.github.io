---
title: "Building Your First Docker Image (part 2 of 6)"
date: 2025-10-31
draft: false
description: "Learn how to build custom Docker images using Dockerfiles. Master essential instructions like FROM, RUN, and CMD, understand image layering for efficiency, and explore multi-stage builds."
summary: "This article dives into the heart of Docker: the Image. We'll cover the Dockerfile structure, key commands, the concept of image layers and caching, and practical techniques to optimize your image size and build speed."
tags: ["docker", "dockerfile", "containers", "image optimization"]
series: ["Docker and Docker-compose Miniseries"]
weight: 2
ShowToc: true
TocOpen: true
math: true
cover:
    image: /blogs/cloud-engineering-gcp/docker-overview/docker.svg

---



| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 30-45 minutes |
| **Technical Level** | Beginner |
| **Prerequisites** | Basic command-line knowledge, Docker installed |


---

This is the second part out of 6 in this miniseries. See the table below for the topics that are covered.


| # | Blog Title | Key Topics Covered |
|---|------------|-------------------|
| 1 | Docker Fundamentals: Containers vs VMs | What is Docker • Containers vs VMs • Docker architecture • Images, Containers, Registries • Installation • Basic commands (`run`, `ps`, `images`, `stop`, `rm`) • Container lifecycle |
| **2** | **Building Your First Docker Image** | **Dockerfile basics • Instructions (FROM, RUN, COPY, CMD, ENTRYPOINT, EXPOSE) • Building images • Image layers & caching • Tagging • Multi-stage builds intro • .dockerignore • Docker Hub** |
| 3 | Docker Networking & Storage | Network fundamentals • Network types (bridge, host, none, overlay) • Port mapping • Container communication • Volumes vs bind mounts • Data persistence • Volume management |
| 4 | Docker Compose Essentials | Why Compose • docker-compose.yml structure • Services, networks, volumes • Compose commands (`up`, `down`, `ps`, `logs`, `exec`) • Multi-container apps • Container naming & discovery |
| 5 | Advanced Compose: Dependencies & Environment Management | Service dependencies • Health checks • Environment variables • .env files • Docker secrets • Compose profiles • Scaling services • Configuration overrides |
| 6 | Docker Best Practices & Optimization | Image optimization • Security best practices • Multi-stage builds deep dive • Logging strategies • Resource limits • Restart policies • Monitoring basics • Common pitfalls • Next steps |



## Introduction: From Consumer to Creator

In the first blog, you learned how to run containers from existing images. Now it's time to level up—let's build your own Docker images from scratch.

Creating custom Docker images gives you complete control over your application environment. Instead of relying on generic images, you can craft precisely what you need: your code, your dependencies, your configuration. This is where Docker truly shines for application deployment.

By the end of this guide, you'll understand how to write Dockerfiles, build images efficiently, and apply best practices that separate hobbyist containers from production-ready deployments.

## What is a Dockerfile?

A **Dockerfile** is a text document containing all the commands needed to build a Docker image. Think of it as a recipe that Docker follows step-by-step to create your container image.

When you run `docker build`, Docker reads your Dockerfile and executes each instruction in order, creating a new image layer for each step. The result is a complete, reproducible image that anyone can build identically.

### Basic Dockerfile Structure

Here's what a simple Dockerfile looks like:

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

Let's break down what's happening here before we dive into each instruction in detail.

## Dockerfile Instructions: The Building Blocks

Every Dockerfile instruction serves a specific purpose. Let's explore the most important ones you'll use daily.

### FROM - Setting the Base Image

**Syntax:** `FROM <image>[:<tag>] [AS <name>]`

The `FROM` instruction sets the base image for your build. It must be the first instruction in your Dockerfile (except for parser directives and `ARG` before `FROM`).

```dockerfile
# Use official Node.js 18 on Alpine Linux
FROM node:18-alpine

# Use specific Ubuntu version
FROM ubuntu:22.04

# Use official Python image
FROM python:3.11-slim
```

**Best practices:**
- Use official images from Docker Hub when possible
- Specify exact versions with tags instead of `latest` for reproducibility
- Prefer slim or alpine variants for smaller image sizes
- Always use trusted sources to avoid security vulnerabilities

### WORKDIR - Setting the Working Directory

**Syntax:** `WORKDIR /path/to/directory`

`WORKDIR` sets the working directory for subsequent instructions. If the directory doesn't exist, Docker creates it automatically.

```dockerfile
WORKDIR /app
```

All future `RUN`, `CMD`, `ENTRYPOINT`, `COPY`, and `ADD` instructions will execute from this directory.

**Best practices:**
- Always use absolute paths for clarity
- Use `WORKDIR` instead of `RUN cd /some/path` for better readability
- Create a dedicated application directory like `/app` or `/usr/src/app`

### COPY - Copying Files into the Image

**Syntax:** `COPY <source> <destination>`

`COPY` copies files and directories from your build context into the container's filesystem.

```dockerfile
# Copy a single file
COPY package.json /app/

# Copy multiple files
COPY package.json package-lock.json ./

# Copy entire directory
COPY src/ /app/src/

# Copy everything from current directory
COPY . .
```

**Best practices:**
- Copy only what you need to keep images small
- Use specific file patterns instead of copying everything
- Copy dependency files first and install them, then do the other copies to leverage Docker's layer caching
- Paths are relative to the build context (usually your current directory)

### ADD - Advanced File Copying

**Syntax:** `ADD <source> <destination>`

`ADD` is similar to `COPY` but with additional features:
- Can extract tar archives automatically
- Can download files from URLs

```dockerfile
# Extract tar archive
ADD application.tar.gz /app/

# Download from URL (not recommended)
ADD https://example.com/file.txt /app/
```

**Best practices:**
- Prefer `COPY` over `ADD` unless you need the extra functionality
- Using `ADD` with URLs is discouraged; use `RUN curl` or `RUN wget` instead for better control

### RUN - Executing Commands

**Syntax:** 
- Shell form: `RUN <command>`: The shell form (`RUN <command>`) executes commands inside a standard shell (like /bin/sh), allowing for shell features such as command chaining with `&&`, piping `|`, and automatic environment variable expansion. This form is often used in Dockerfiles when multiple commands need to be run efficiently within a single layer. The primary downside is that the shell process receives the OS signals, which can complicate graceful shutdown behavior for the container's main process.
- Exec form: `RUN ["executable", "param1", "param2"]`: The exec form (`RUN ["executable", "param1", "param2"]`), conversely, executes the command directly without an intervening shell. This approach prevents shell features like && or automatic variable expansion but offers cleaner execution, better consistency across different environments, and improved signal handling, as the executable is the main process (PID 1). To use shell features with the exec form, one must explicitly invoke the shell, such as `RUN ["sh", "-c", "command && command"]`

`RUN` executes commands during the build process. Each `RUN` instruction creates a new layer in the image.

```dockerfile
# Install packages (Debian/Ubuntu)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js dependencies
RUN npm install --production

# Multiple commands in one RUN
RUN npm install && \
    npm run build && \
    npm prune --production
```

**Best practices:**
- Chain commands with `&&` to reduce layers
- Clean up in the same layer (like removing apt cache)
- Use backslashes for multi-line commands for readability
- Minimize the number of `RUN` instructions to reduce layer count

### ENV - Setting Environment Variables

**Syntax:** `ENV <key>=<value>`

`ENV` sets environment variables that persist both during build and in running containers.

```dockerfile
# Single variable
ENV NODE_ENV=production

# Multiple variables
ENV APP_HOME=/app \
    PORT=3000 \
    LOG_LEVEL=info
```

These variables are available in all subsequent instructions and in the running container.

### EXPOSE - Documenting Ports

**Syntax:** `EXPOSE <port> [<port>/<protocol>...]`

`EXPOSE` documents which ports your application listens on. It doesn't actually publish the port—that happens with `docker run -p`.

```dockerfile
# Expose HTTP port
EXPOSE 3000

# Expose multiple ports
EXPOSE 3000 8080

# Specify protocol
EXPOSE 3000/tcp
EXPOSE 5353/udp
```

**Best practices:**
- Always document exposed ports for clarity
- Use standard ports when possible (80 for HTTP, 443 for HTTPS)

### CMD - Default Command

**Syntax:**
- Exec form (preferred): `CMD ["executable", "param1", "param2"]`
- Shell form: `CMD command param1 param2`

`CMD` specifies the default command to run when a container starts. Only the last `CMD` in a Dockerfile takes effect.

```dockerfile
# Start Node.js application
CMD ["node", "server.js"]

# Start with npm
CMD ["npm", "start"]

# Shell form (not recommended)
CMD node server.js
```

**Key point:** `CMD` can be overridden when running the container:
```bash
docker run myimage python app.py  # Overrides CMD
```

### ENTRYPOINT - Main Executable

**Syntax:**
- Exec form (preferred): `ENTRYPOINT ["executable", "param1"]`
- Shell form: `ENTRYPOINT command param1`

`ENTRYPOINT` sets the main command that always runs. Unlike `CMD`, it's harder to override.

```dockerfile
# Container always runs node
ENTRYPOINT ["node"]

# Combined with CMD for default arguments
ENTRYPOINT ["node"]
CMD ["server.js"]
```

When both `ENTRYPOINT` and `CMD` are present, `CMD` provides default arguments to `ENTRYPOINT`.

### CMD vs ENTRYPOINT: When to Use Which?

Use `CMD` when:
- You want users to easily override the command
- The container can run different commands

Use `ENTRYPOINT` when:
- Your container is a tool or specific application
- You want to enforce a specific executable

Use both when:
- You want a fixed command with overridable default arguments


---


## Writing Your First Dockerfile: Simple Python Flask App

Let's create a simple Python Flask web application and containerize it.

### Step 1: Create the Application

First, create a project directory with these files:

**requirements.txt:**
```text
Flask==3.0.0
```

**server.py:**
```python
from flask import Flask, jsonify
from datetime import datetime
import os

app = Flask(__name__)
PORT = int(os.environ.get('PORT', 3000))

@app.route('/')
def home():
    return jsonify({
        'message': 'Hello from Docker!',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print(f'Server running on port {PORT}')
    app.run(host='0.0.0.0', port=PORT)
```

### Step 2: Write a Simple Dockerfile

**Dockerfile:**
```dockerfile
# Use official Python 3.13 on Alpine Linux (smaller size)
FROM python:3.13-alpine

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py ./

# Expose the port
EXPOSE 3000

# Set environment variable
ENV FLASK_ENV=production

# Start the application
CMD ["python", "server.py"]
```

This is a simple, straightforward Dockerfile that gets the job done.

## Building Images: The docker build Command

Now let's build our image.

### Basic Build Command

```bash
docker build -t my-flask-app:1.0 .
```

Let's break this down:
- `docker build` - The build command
- `-t my-flask-app:1.0` - Tag the image with name and version
- `.` - Build context (current directory)

### Build Process

When you run this command, Docker:

1. Sends the build context to the Docker daemon
2. Executes each Dockerfile instruction in order
3. Creates a new layer for each instruction
4. Caches layers for faster subsequent builds
5. Tags the final image

You'll see output like this:

```
[+] Building 21.6s (10/10) FINISHED                              docker:default
 => [internal] load build definition from Dockerfile                       0.1s
 => => transferring dockerfile: 466B                                       0.0s
 => [internal] load metadata for docker.io/library/python:3.13-alpine      7.4s
 => [internal] load .dockerignore                                          0.0s
 => => transferring context: 2B                                            0.0s
 => [1/5] FROM docker.io/library/python:3.13-alpine@sha256:e5fa639e49b859  4.5s
 => => resolve docker.io/library/python:3.13-alpine@sha256:e5fa639e49b859  0.0s
 ...
 => [internal] load build context                                          0.0s
 => => transferring context: 560B                                          0.0s
 => [2/5] WORKDIR /app                                                     0.2s
 => [3/5] COPY requirements.txt ./                                         0.1s
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt               8.4s
 => [5/5] COPY server.py ./                                                0.2s
 => exporting to image                                                     0.4s
 => => exporting layers                                                    0.3s
 => => writing image sha256:bb22f69656e8f75dcd9ef2c0c406ea9fea57d20957743  0.0s
 => => naming to docker.io/library/my-flask-app:1.0                        0.0s
```

### Useful Build Options

```bash
# Build without cache (fresh build)
docker build --no-cache -t my-app .

# Build with different Dockerfile name
docker build -f Dockerfile.prod -t my-app .

# Build with build arguments
docker build --build-arg PYTHON_VERSION=3.13 -t my-app .

# Build and show detailed output
docker build --progress=plain -t my-app .
```

---

## Understanding Image Layers and Caching

Docker images are built in layers. Each instruction in your Dockerfile creates a new layer.

### How Layers Work

```dockerfile
FROM python:3.13-alpine              # Layer 1: Base image
WORKDIR /app                         # Layer 2: Create /app directory
COPY requirements.txt ./             # Layer 3: Copy requirements file
RUN pip install -r requirements.txt  # Layer 4: Install dependencies
COPY . .                             # Layer 5: Copy application code
CMD ["python", "server.py"]          # Layer 6: Set default command
```

Each layer is cached. If nothing changes in a layer, Docker reuses the cached version.

### Optimizing for Cache

**Bad example** (poor caching):
```dockerfile
FROM python:3.13-alpine
WORKDIR /app
COPY . .                           # Copies everything
RUN pip install -r requirements.txt  # Reinstalls every time code changes
CMD ["python", "server.py"]
```

**Good example** (optimized caching):
```dockerfile
FROM python:3.13-alpine
WORKDIR /app
COPY requirements.txt ./           # Copy deps first
RUN pip install -r requirements.txt  # Cached unless requirements change
COPY . .                           # Copy code last
CMD ["python", "server.py"]
```

The second approach is much faster because `pip install` only runs when dependencies actually change, not every time you modify your code.


## Tagging Images Properly

Tags help you organize and version your images.

### Tag Syntax

```
[registry/][username/]image-name[:tag]
```

### Common Tagging Patterns

```bash
# Version tags
docker build -t my-app:1.0.0 .
docker build -t my-app:1.0 .
docker build -t my-app:1 .

# Environment tags
docker build -t my-app:dev .
docker build -t my-app:staging .
docker build -t my-app:prod .

# Multiple tags for same image
docker build -t my-app:1.0.0 -t my-app:latest .

# With registry
docker build -t myregistry.com/my-app:1.0.0 .
```

### Best Practices for Tags

1. **Never rely on `latest` in production** - it's mutable and unpredictable
2. **Use semantic versioning** - `1.2.3` format
3. **Include git commit SHA** - for traceability: `my-app:abc123f`
4. **Tag for different purposes** - version, environment, commit


--- 

## Code Demo: Comprehensive Dockerfile

Now let's build a more comprehensive Dockerfile that showcases various functionalities:

**Dockerfile.comprehensive:**
```dockerfile
# syntax=docker/dockerfile:1

# Build argument for Python version
ARG PYTHON_VERSION=3.13

# Use official Python image with specific version
FROM python:${PYTHON_VERSION}-alpine AS base

# Maintainer label
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Comprehensive Python application"

# Install system dependencies
RUN apk add --no-cache \
    curl \
    tini

# Create app user for security (don't run as root)
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001 -G appuser

# Set working directory
WORKDIR /app

# Copy requirements file with proper ownership
COPY --chown=appuser:appuser requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directory for logs
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs

# Set environment variables
ENV FLASK_ENV=production \
    PORT=3000 \
    LOG_DIR=/app/logs \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 3000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python healthcheck.py || exit 1

# Switch to non-root user
USER appuser

# Use tini as init system (handles signals properly)
ENTRYPOINT ["/sbin/tini", "--"]

# Start application
CMD ["python", "server.py"]
```

**healthcheck.py:**
```python
import http.client
import os
import sys

def check_health():
    try:
        port = int(os.environ.get('PORT', 3000))
        conn = http.client.HTTPConnection('localhost', port, timeout=2)
        conn.request('GET', '/health')
        response = conn.getresponse()
        
        if response.status == 200:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception:
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    check_health()
```

This comprehensive Dockerfile includes:
- Build arguments for flexibility
- Labels for metadata
- System dependencies
- Security (non-root user)
- Proper dependency management
- Health checks
- Signal handling with tini
- Production-ready configuration


## Understanding the Comprehensive Dockerfile

Before we build and run this Dockerfile, let's break down each section and understand the choices we've made. This comprehensive Dockerfile showcases production-ready practices that go far beyond a basic setup.

| Dockerfile Section | What It Does | Why It Matters |
|-------------------|--------------|----------------|
| **Syntax Directive**<br>`# syntax=docker/dockerfile:1` | Uses the latest Dockerfile syntax | Access to modern features and improvements |
| **Build Arguments**<br>`ARG PYTHON_VERSION=3.13` | Defines build-time variable for Python version | Flexibility to build with different versions without editing:<br>`docker build --build-arg PYTHON_VERSION=3.12 -t my-app .` |
| **Base Image**<br>`FROM python:${PYTHON_VERSION}-alpine AS base` | Uses official Python on Alpine Linux | **Alpine benefits**: 5-10x smaller images, reduced attack surface, faster deployments<br>**Trade-off**: Uses musl libc; some packages may need `python:3.13-slim` instead |
| **Labels**<br>`LABEL maintainer="..."`<br>`LABEL version="1.0"` | Adds metadata to the image | Helps identify images, enables automation, provides documentation |
| **System Dependencies**<br>`RUN apk add --no-cache curl tini` | Installs curl (debugging/health checks) and tini (init system)<br>`--no-cache` prevents storing package index | Reduces image size; tini enables proper signal handling and graceful shutdown |
| **Non-Root User**<br>`RUN addgroup -g 1001 -S appuser &&`<br>`adduser -S appuser -u 1001 -G appuser` | Creates system user/group with specific UID/GID 1001 | **Critical security**: Running as root means vulnerabilities could compromise the host<br>Specific UID/GID ensures consistent permissions across systems |
| **Working Directory**<br>`WORKDIR /app` | Creates and sets `/app` as working directory | Convention, keeps app files separate from system files |
| **Dependencies**<br>`COPY --chown=appuser:appuser requirements.txt ./`<br>`RUN pip install --no-cache-dir -r requirements.txt` | Copies requirements with correct ownership, installs packages | **Copy requirements first**: Leverages layer caching (dependencies rarely change)<br>`--no-cache-dir` saves 100+ MB by not caching packages |
| **Application Code**<br>`COPY --chown=appuser:appuser . .` | Copies all application files with correct ownership | **Copied last**: Code changes frequently; copying last maximizes cache hits |
| **Log Directory**<br>`RUN mkdir -p /app/logs &&`<br>`chown -R appuser:appuser /app/logs` | Creates logs directory with proper permissions | Ensures non-root user can write logs (prefer stdout/stderr in production) |
| **Environment Variables**<br>`ENV FLASK_ENV=production`<br>`ENV PORT=3000`<br>`ENV PYTHONUNBUFFERED=1` | Sets Flask environment, port, and Python output mode | **PYTHONUNBUFFERED=1 is critical**: Forces immediate log output; without it, logs are buffered and debugging becomes painful |
| **Port Documentation**<br>`EXPOSE 3000` | Documents that app listens on port 3000 | Doesn't publish port (use `docker run -p`), but helps developers and orchestrators |
| **Health Check**<br>`HEALTHCHECK --interval=30s --timeout=3s`<br>`--start-period=5s --retries=3`<br>`CMD python healthcheck.py \|\| exit 1` | Runs health check every 30s, allows 3s per check, 5s startup grace, 3 retries before unhealthy | Kubernetes/Swarm use this to restart containers; load balancers remove unhealthy containers; provides built-in monitoring |
| **User Switch**<br>`USER appuser` | Switches from root to appuser for all subsequent commands | **Must come after root operations**; application runs as non-root for security |
| **Init System**<br>`ENTRYPOINT ["/sbin/tini", "--"]` | Makes tini PID 1, which starts the app as child process | **Why needed**: Forwards signals properly, reaps zombie processes, enables graceful shutdown<br>Without tini: `docker stop` takes 10s then SIGKILL; with tini: graceful shutdown in 1-2s |
| **Application Start**<br>`CMD ["python", "server.py"]` | Default command to run (becomes argument to tini) | Can be overridden at runtime: `docker run my-app python manage.py migrate` |

### Design Principles Summary

| Principle | Implementation | Benefit |
|-----------|---------------|---------|
| **Security First** | Non-root user, minimal Alpine base, no unnecessary tools | Reduced attack surface, compliance-ready |
| **Observability** | Health checks, proper logging with PYTHONUNBUFFERED | Automated recovery, real-time debugging |
| **Maintainability** | Build arguments, labels, clear structure | Easy updates, good documentation |
| **Production Ready** | Tini for signal handling, graceful shutdown, optimized caching | Reliable deployments, zero-downtime updates |
| **Size Optimization** | Alpine base, --no-cache flags, smart layering | 5x smaller images (~80-100MB vs ~500MB), faster deployments |


-----

## Building and Running the Comprehensive Dockerfile

Now that we have our comprehensive Dockerfile, let's build and run it to see everything in action.

### Step 1: Build the Image

Build the image with the comprehensive Dockerfile:

```bash
docker build -f Dockerfile.comprehensive -t my-python-app:comprehensive .
```

**Expected output:**

```
[+] Building 17.2s (15/15) FINISHED                              docker:default
 => [internal] load build definition from Dockerfile.comprehensive         0.0s
 => => transferring dockerfile: 1.40kB                                     0.0s
 => resolve image config for docker-image://docker.io/docker/dockerfile:1  0.5s
 => CACHED docker-image://docker.io/docker/dockerfile:1@sha256:b6afd42430  0.0s
 => [internal] load metadata for docker.io/library/python:3.13-alpine      0.5s
 => [internal] load .dockerignore                                          0.0s
 => => transferring context: 2B                                            0.0s
 => [1/8] FROM docker.io/library/python:3.13-alpine@sha256:e5fa639e49b859  0.0s
 => [internal] load build context                                          0.0s
 => => transferring context: 198B                                          0.0s
 => CACHED [2/8] RUN apk add --no-cache     curl     tini                  0.0s
 => CACHED [3/8] RUN addgroup -g 1001 -S appuser &&     adduser -S appuse  0.0s
 => CACHED [4/8] WORKDIR /app                                              0.0s
 => [5/8] COPY --chown=appuser:appuser requirements.txt ./                 0.1s
 => [6/8] RUN pip install --no-cache-dir -r requirements.txt              13.7s
 => [7/8] COPY --chown=appuser:appuser . .                                 0.3s 
 => [8/8] RUN mkdir -p /app/logs &&     chown -R appuser:appuser /app/log  0.7s 
 => exporting to image                                                     0.7s 
 => => exporting layers                                                    0.7s 
 => => writing image sha256:ad65a05cff289a53f0c3ae5fa3ad60620618f261349c2  0.0s 
 => => naming to docker.io/library/my-python-app:comprehensive             0.0s
```

The build process shows each layer being created. Notice how Docker caches layers—if you rebuild without changes, it will be much faster.

### Step 2: Run the Container

Start the container in detached mode:

```bash
docker run -d -p 3000:3000 --name my-app my-python-app:comprehensive
```

**What this command does:**
- `-d` - Run in detached mode (background)
- `-p 3000:3000` - Map port 3000 on host to port 3000 in container
- `--name my-app` - Give the container a friendly name
- `my-python-app:comprehensive` - The image to run

**Expected output:**
```
a hashcode like this: f8a3d9b2c1e4567890abcdef1234567890abcdef1234567890abcdef12345678
```

This is the container ID, confirming it's running.

### Step 3: Verify the Container is Running

Check the container status:

```bash
docker ps
```

**Expected output:**

```
CONTAINER ID   IMAGE                          COMMAND                  CREATED          STATUS                    PORTS                    NAMES
f8a3d9b2c1e4   my-python-app:comprehensive    "/sbin/tini -- pytho…"   10 seconds ago   Up 8 seconds (healthy)    0.0.0.0:3000->3000/tcp   my-app
```

**Key observations:**
- **STATUS** shows `(healthy)` - The health check is working!
- **COMMAND** shows `/sbin/tini --` - Tini is handling process signals
- **PORTS** shows the port mapping

### Step 4: View Container Logs

Check what's happening inside:

```bash
docker logs my-app
```

**Expected output:**

```
Server running on port 3000
 * Serving Flask app 'server'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:3000
 * Running on http://172.17.0.2:3000
```

The application is running and listening on port 3000.

### Step 5: Test the Application

Test the main endpoint:

```bash
curl http://localhost:3000/
```

**Expected output:**

```json
{
  "message": "Hello from Docker!",
  "timestamp": "2025-10-31T14:23:45.123456Z"
}
```

Test the health endpoint:

```bash
curl http://localhost:3000/health
```

**Expected output:**

```json
{
  "status": "healthy"
}
```

### Step 6: Inspect the Health Check

Docker automatically runs health checks. View the health status:

```bash
docker inspect --format='{{json .State.Health}}' my-app | python -m json.tool
```

**Expected output:**

```json
{
  "Status": "healthy",
  "FailingStreak": 0,
  "Log": [
    {
      "Start": "2025-10-31T14:23:30.123456789Z",
      "End": "2025-10-31T14:23:30.456789012Z",
      "ExitCode": 0,
      "Output": ""
    }
  ]
}
```

**What this shows:**
- Health checks run every 30 seconds (as configured)
- `ExitCode: 0` means the check passed
- Docker tracks the health check history

### Step 7: Check Container Resource Usage

See how much resources the container is using:

```bash
docker stats my-app --no-stream
```

**Expected output:**

```
CONTAINER ID   NAME      CPU %     MEM USAGE / LIMIT     MEM %     NET I/O          BLOCK I/O   PIDS
932664ac60a3   my-app    0.02%     21.73MiB / 7.639GiB   0.28%     4.85kB / 1.4kB   0B / 0B     2
```

**Key metrics:**
- **CPU %** - Very low for an idle web server
- **MEM USAGE** - Around 21MB (Alpine base keeps it small)
- **PIDS** - 2 processes (tini + Python Flask worker)

### Step 8: Execute Commands Inside the Container

Open a shell inside the running container:

```bash
docker exec -it my-app sh
```

Once inside, you can explore:

```bash
# Check the user (should be 'appuser', not root)
whoami
# Output: appuser

# List files (owned by appuser)
ls -la
# Output shows files owned by appuser:appuser

# Check Python version
python --version
# Output: Python 3.13.x

# Check installed packages
pip list
# Output: Flask and its dependencies

# Exit the container
exit
```

### Step 9: Monitor Logs in Real-Time

Follow the logs as they happen:

```bash
docker logs -f my-app
```

In another terminal, make some requests:

```bash
curl http://localhost:3000/
curl http://localhost:3000/health
```

You'll see log entries appear in real-time in the first terminal showing incoming requests.

Press `Ctrl+C` to stop following logs.

### Step 10: Test Signal Handling (Graceful Shutdown)

Stop the container gracefully:

```bash
docker stop my-app
```

**What happens:**
1. Docker sends SIGTERM signal to tini
2. Tini forwards the signal to the Python process
3. Flask shuts down gracefully
4. Container exits cleanly

**Expected output:**
```
my-app
```

Check it's stopped:

```bash
docker ps -a | grep my-app
```

**Expected output:**

```
CONTAINER ID   IMAGE                          COMMAND                  CREATED         STATUS                      PORTS     NAMES
f8a3d9b2c1e4   my-python-app:comprehensive    "/sbin/tini -- pytho…"   5 minutes ago   Exited (143) 10 seconds ago             my-app
```

### Step 11: Restart the Container

Restart the stopped container:

```bash
docker start my-app
```

The container resumes with all its configuration intact.

### Step 12: Clean Up

When you're done testing:

```bash
# Stop the container
docker stop my-app

# Remove the container
docker rm my-app

# Remove the image (optional)
docker rmi my-python-app:comprehensive
```

## What Makes This Dockerfile Comprehensive?

Let's review what we demonstrated:

1. **Security**: The application runs as a non-root user (`appuser`), reducing attack surface if the container is compromised.

2. **Health Monitoring**: Automatic health checks every 30 seconds ensure Docker knows if your app is truly running or just hanging.

3. **Signal Handling**: Tini ensures proper process cleanup and graceful shutdowns when stopping containers.

4. **Small Image Size**: Using Alpine Linux and `--no-cache-dir` with pip keeps the image under 100MB.

5. **Production Configuration**: Environment variables are properly set, including `PYTHONUNBUFFERED=1` for immediate log output.

6. **Proper Ownership**: All files are owned by the application user, not root.

7. **Build Arguments**: The Python version can be changed at build time without editing the Dockerfile.

8. **Labels**: Metadata helps identify and organize images in larger deployments.

This comprehensive approach demonstrates production-ready practices that you should apply to real-world applications.


-----

## Multi-Stage Builds Basics

Multi-stage builds let you use multiple `FROM` statements in one Dockerfile, dramatically reducing image size.

### Why Multi-Stage Builds?

Traditional builds include everything: build tools, source code, dependencies. Multi-stage builds:
1. Build in one stage (with all build tools)
2. Copy only necessary artifacts to a minimal runtime stage
3. Result in 70-80% smaller final images

### Simple Multi-Stage Example

```dockerfile
# Stage 1: Build stage
FROM python:3.13-alpine AS builder

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m compileall .

# Stage 2: Production stage
FROM python:3.13-alpine AS production

WORKDIR /app

# Copy only runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy compiled code from builder stage
COPY --from=builder /app ./

EXPOSE 3000

CMD ["python", "server.py"]
```

**How It Works:**
- **builder stage**: Has all dependencies, compiles Python code
- **production stage**: Starts fresh, only copies what's needed to run
- `COPY --from=builder`: Copies files from builder stage
- Final image only contains production stage—builder is discarded!

### Real-World Multi-Stage Build

Here's a practical example for a Flask application with frontend assets:

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Build frontend assets
FROM node:18-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python dependencies
FROM python:3.13-alpine AS python-builder

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 3: Production image
FROM python:3.13-alpine AS production

WORKDIR /app

# Copy Python packages from builder
COPY --from=python-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Copy built frontend assets
COPY --from=frontend-builder /frontend/dist ./static

# Create non-root user
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001 -G appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 3000

CMD ["python", "server.py"]
```

**Benefits:**
- Builds frontend and backend in isolation
- Final image contains only runtime essentials
- Results in images 70-80% smaller than single-stage

## The .dockerignore File

The `.dockerignore` file excludes files from the build context, similar to `.gitignore`.

### Why Use .dockerignore?

1. **Faster builds** - Less data sent to Docker daemon
2. **Smaller images** - Prevents unnecessary files from being copied
3. **Security** - Keeps sensitive files out of images
4. **Better caching** - Avoids cache invalidation from irrelevant changes

### Python-Specific .dockerignore

Create `.dockerignore` in your project root:

```
# Version control
.git
.gitignore
.github/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# Environment files
.env
.env.*
*.local

# Testing
.pytest_cache/
.coverage
htmlcov/
*.test.py

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Documentation
README.md
docs/
*.md

# Docker files
Dockerfile*
docker-compose*.yml
.dockerignore
```

### Advanced Patterns

```
# Ignore everything
*

# Except these
!requirements.txt
!app/
!server.py

# Ignore specific patterns
**/*.log
**/__pycache__/
**/temp/

# Ignore except specific files
*.md
!README.md
```

## Pushing Images to Docker Hub

Share your images via Docker Hub in four simple steps.

### Steps to Push

```bash
# 1. Create account at hub.docker.com

# 2. Login
docker login

# 3. Tag image with your username
docker tag my-python-app:1.0 yourusername/my-python-app:1.0

# 4. Push to Docker Hub
docker push yourusername/my-python-app:1.0
```

### Push Multiple Tags

```bash
docker push yourusername/my-python-app:1.0
docker push yourusername/my-python-app:latest
```

### Pull Your Image

Anyone can now pull and run your image:

```bash
docker pull yourusername/my-python-app:1.0
docker run -p 3000:3000 yourusername/my-python-app:1.0
```

Visit `https://hub.docker.com/r/yourusername/my-python-app` to view your published image.

## Key Takeaways

1. **Dockerfiles are recipes** - Each instruction creates a layer. Order matters for caching efficiency.

2. **Essential instructions** - `FROM` (base), `WORKDIR` (directory), `COPY` (files), `RUN` (commands), `ENV` (environment), `EXPOSE` (ports), `CMD`/`ENTRYPOINT` (startup).

3. **Layer caching optimization** - Copy dependency files before source code to maximize cache hits.

4. **Proper tagging** - Use semantic versioning (1.2.3), never rely on `latest` in production.

5. **Multi-stage builds** - Build in one stage, copy only artifacts to runtime stage. Reduces image size by 70-80%.

6. **.dockerignore file** - Excludes unnecessary files (`__pycache__`, logs, `.env`) for faster builds and better security.

7. **Security best practices** - Non-root users, health checks, specific versions, no secrets in images.

8. **Docker Hub** - Share images publicly or privately. Tag with username, push, and deploy anywhere.

In the next blog, we'll explore Docker networking and storage—how containers communicate and persist data beyond container lifecycles.


---


**Series Navigation**:
- [Part 1: Docker Fundamentals - Containers vs VMs](../p1)
- **Part 2: Building Your First Docker Image** (You are here)
- Part 3: Docker Networking & Storage
- Part 4: Docker Compose Essentials
- Part 5: Advanced Compose: Dependencies & Environment Management
- Part 6: Docker Best Practices & Optimization

**Next in Series**: [Docker Networking & Storage](../p3) - Learn how containers communicate, understand Docker network types, and master data persistence with volumes.
