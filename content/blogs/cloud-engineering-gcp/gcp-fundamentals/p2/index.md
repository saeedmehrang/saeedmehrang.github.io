---
title: "Google Cloud Fundamentals: Learning Notes Summary (Part 2 of 4)"
date: 2025-10-28
draft: false
author: Saeed Mehrang
description: "A deep dive into GCP compute and networking: Compute Engine, VPC networking, load balancing, autoscaling, VPC peering, Cloud DNS, CDN, and connecting to external networks."
summary: "Explore GCP's compute and networking capabilities: Virtual Private Cloud architecture, Compute Engine VM types and pricing, autoscaling, load balancing options, and multiple ways to connect VPCs to external networks."
tags: ["GCP", "Google Cloud", "Compute Engine", "VPC", "Networking", "Load Balancing", "Cloud Infrastructure"]
categories: ["Cloud", "Tutorial"]
series_order: 2
showToc: true
disableAnchoredHeadings: false
cover:
    image: /blogs/cloud-engineering-gcp/gcp-fundamentals/Google_Cloud_logo.svg
---


> **Course:** Google Cloud Fundamentals: Core Infrastructure | Google Cloud Skills Boost   https://www.cloudskillsboost.google
> **Focus:** Virtual Machines and Networks in the Cloud

---

## Introduction

This is the second part of my systematic GCP learning journey. While this article stands alone, I recommend reading [Part 1](link-to-part-1) for foundational concepts like resource hierarchy and IAM.

Part 2 focuses on compute and networking—the building blocks of cloud infrastructure.

## What's Covered

This article explores GCP's compute and networking capabilities:

- Virtual Private Cloud (VPC) fundamentals
- Compute Engine as IaaS
- VM types and pricing models
- Autoscaling strategies
- VPC networking features
- VPC connectivity (peering, shared VPC)
- Load balancing options
- DNS and edge caching
- Connecting VPCs to external networks

---

## Virtual Private Cloud (VPC): Your Private Space in the Public Cloud

A **Virtual Private Cloud (VPC)** is your own isolated network environment within Google Cloud. Think of it as having a private cloud with all its benefits—security, control, customization—but hosted on Google's public infrastructure.

### What Makes Google Cloud VPC Special

**Global Scope**: Unlike traditional networks, Google Cloud VPCs span the entire globe by default.

**Regional Subnets**: You can segment your VPC into subnets, each located in any Google Cloud region worldwide.

**Cross-Zone Flexibility**: Subnets can extend across multiple zones within a region, meaning resources in different zones can belong to the same subnet.

```
VPC (Global)
├── Subnet: us-central1 (Regional)
│   ├── Zone: us-central1-a (Resources here)
│   └── Zone: us-central1-b (Resources here)
└── Subnet: europe-west1 (Regional)
    ├── Zone: europe-west1-a
    └── Zone: europe-west1-b
```

This architecture simplifies creating global applications without complex network configurations.

---

## Compute Engine: Google's IaaS Solution

Compute Engine lets you create and run virtual machines on Google's infrastructure with the performance and functionality of physical servers.

### Key Features

- **No upfront investment**: Pay only for what you use
- **Massive scale**: Run thousands of virtual CPUs
- **Full OS control**: Configure CPU, memory, storage, and operating system
- **Consistent performance**: Built on Google's reliable infrastructure

### Creating Virtual Machines

You can create VMs through:
- Google Cloud Console (web interface)
- Google Cloud CLI (command line)
- Compute Engine API (programmatic access)

### Operating System Options

- Linux and Windows Server images from Google
- Customized versions of provided images
- Your own operating system images

**Quick Start Tip**: Use the Cloud Marketplace for pre-configured solutions from Google and third-party vendors. Note that third-party images may include additional licensing costs for commercial software.

---

## Compute Engine Pricing: Understanding Your Costs

Google's pricing model is designed for flexibility and cost savings:

| Pricing Feature | Details | Savings |
|-----------------|---------|---------|
| **Billing Granularity** | Per-second billing (1-minute minimum) | Pay only for actual usage |
| **Sustained-Use Discounts** | Automatic discounts for VMs running >25% of month | Automatic savings |
| **Committed-Use Discounts** | 1-year or 3-year commitment | Up to 57% off for predictable workloads |

---

## Three Types of Compute Engine VMs

Choose the VM type based on your workload characteristics:

### Persistent (On-Demand) VMs

**Use case**: Standard workloads requiring continuous availability

- Run as long as you need
- Full control over termination
- No interruptions
- Standard pricing

### Preemptible VMs

**Use case**: Batch jobs, fault-tolerant workloads, development/testing

- **Maximum runtime**: 24 hours
- **Cost savings**: 60-91% discount
- **Limitation**: Google can stop them at any time
- **Ideal for**: Jobs that can pause and resume

**Example scenarios**: 
- Data processing pipelines
- Rendering tasks
- Scientific computations
- CI/CD test environments

### Spot VMs

**Use case**: Same as Preemptible, but with more flexibility

- **Same pricing** as Preemptible VMs (currently)
- **No maximum runtime** (unless you specify one)
- **Stopped only** when Google needs capacity elsewhere
- **Additional benefits**: Automated workload scheduling, cluster autoscaling

**Key difference**: Spot VMs are the evolved version of Preemptible VMs with enhanced features.

### Custom Machine Types

Beyond standard VM configurations, you can create custom machine types by specifying:
- Exact number of virtual CPUs needed
- Precise amount of memory required

**Note**: Maximum CPUs per VM depend on the machine family and your zone-specific quota.

---

## Autoscaling: Dynamic Resource Management

Autoscaling automatically adjusts your VM count based on demand, ensuring optimal performance and cost efficiency.

### How It Works

```
Low Traffic → Fewer VMs → Lower Costs
    ↓
Traffic Increases → Autoscaling Adds VMs
    ↓
High Traffic → More VMs → Maintained Performance
    ↓
Traffic Decreases → Autoscaling Removes VMs
    ↓
Back to Low Traffic → Cost Optimization
```

### Scaling Strategies

**Scale Out (Horizontal)**: Add more VMs to handle increased load—this is how most Google Cloud customers start.

**Scale Up (Vertical)**: Use larger VMs with more resources—ideal for specific workloads like:
- In-memory databases
- CPU-intensive analytics
- Single-threaded applications

---

## VPC Networking Capabilities: Built-In Intelligence

Google Cloud VPCs come with networking features that you don't need to provision or manage separately.

### Automatic Routing

**No router to configure**: VPCs include built-in routing tables automatically.

**What they do**:
- Forward traffic between instances in the same network
- Route across subnetworks
- Enable communication between Google Cloud zones
- Work without external IP addresses

### Native Firewall Protection

**No firewall appliance needed**: VPCs provide a global distributed firewall out of the box.

**Control options**:
- Restrict incoming traffic (ingress rules)
- Restrict outgoing traffic (egress rules)
- Define rules using network tags on instances
- Apply rules globally or to specific resources

**Convenience**: Network tags make firewall management simple—tag your instances and apply rules to those tags.

---

## Connecting VPCs: Peering and Sharing

### VPC Peering

**Problem**: VPCs are project-specific. How do you connect resources across projects?

**Solution**: VPC Peering establishes private connectivity between VPCs.

```
Project A (VPC-A) ←→ VPC Peering ←→ Project B (VPC-B)
```

**Benefits**:
- Private communication between projects
- No internet exposure
- Lower latency
- Reduced egress costs

### Shared VPC

**Use case**: Organizations with multiple projects needing centralized network management.

**How it works**:

1. Designate one project as the **host project**
2. Attach other **service projects** to it
3. Resources in service projects use the host project's VPC network

```
Host Project (Shared VPC Network)
├── Service Project 1 (uses host VPC)
├── Service Project 2 (uses host VPC)
└── Service Project 3 (uses host VPC)
```

**Advantages**:
- Centralized network administration
- Consistent security policies
- Simplified network architecture
- Internal IP communication across projects

---

## Cloud Load Balancing: Distributing Traffic Intelligently

Load balancing distributes incoming traffic across multiple VM instances, preventing any single instance from becoming overwhelmed.

### Why Google's Load Balancing is Different

| Feature | Benefit |
|---------|---------|
| **Fully managed** | No VMs to manage or scale |
| **Software-defined** | Flexible configuration |
| **Global distribution** | Serve users from nearest location |
| **Auto-scaling** | No pre-warming required |
| **Comprehensive protocols** | HTTP(S), TCP, SSL, UDP |

### Load Balancing Options

Choose based on your traffic type and architecture:

#### For Internet-Facing Traffic

| Load Balancer | Traffic Type | Scope | Use Case |
|---------------|--------------|-------|----------|
| **HTTP(S) Load Balancer** | HTTP/HTTPS | Global | Web applications across regions |
| **Global SSL Proxy** | SSL (non-HTTP) | Global | Secure SSL traffic |
| **Global TCP Proxy** | TCP (non-SSL) | Global | Other TCP traffic |

#### For Internal Traffic

| Load Balancer | Layer | Scope | Use Case |
|---------------|-------|-------|----------|
| **Regional Internal Load Balancer** | Layer 4 | Regional | Traffic between application tiers |
| **Internal HTTPS Load Balancer** | Layer 7 | Regional | Internal HTTP(S) services |

**Example architecture**:
```
Internet Users
    ↓
HTTP(S) Load Balancer (Global)
    ↓
Web Tier (Compute Engine VMs)
    ↓
Internal Load Balancer (Regional)
    ↓
Application Tier (Compute Engine VMs)
```

---

## Cloud DNS: Managed Domain Name System

Cloud DNS is Google's managed DNS service running on the same infrastructure that powers Google's own services.

### Key Features

- **Low latency**: Fast DNS resolution
- **High availability**: 100% uptime SLA
- **Global redundancy**: Served from locations worldwide
- **Cost-effective**: Pay only for what you use
- **Programmable**: Manage via Console, CLI, or API

### Scale

Publish and manage **millions of DNS zones and records** without worrying about infrastructure.

---

## Cloud CDN: Edge Caching for Performance

Cloud CDN (Content Delivery Network) uses edge caching to store content closer to your users, reducing latency and improving performance.

### Benefits

| Benefit | Impact |
|---------|--------|
| **Lower latency** | Users get content from nearby servers |
| **Reduced origin load** | Less traffic to your origin servers |
| **Cost savings** | Lower bandwidth costs |

### Easy Setup

Once HTTP(S) Load Balancing is configured, enabling Cloud CDN is just **one checkbox**.

```
User Request → Nearest Edge Location (cached content) → Fast Response
    ↓ (cache miss)
Origin Server → Edge Cache → User
```

---

## Connecting to External Networks

Google Cloud offers multiple ways to connect your VPC to on-premises networks or other cloud providers:

### Connection Options Compared

| Method | Type | Bandwidth | Use Case |
|--------|------|-----------|----------|
| **VPN** | Encrypted tunnel over internet | Flexible | Secure, cost-effective connectivity |
| **Direct Peering** | Direct connection at Google PoP | Varies | Exchange traffic at 100+ global locations |
| **Dedicated Interconnect** | Private physical connection | 10/100 Gbps | High bandwidth, low latency requirements |
| **Partner Interconnect** | Connection via service provider | 50 Mbps - 50 Gbps | Locations not near Google facilities |

### Choosing the Right Option

**VPN (Virtual Private Network)**
- Uses IPsec protocol
- Encrypted tunnel over public internet
- Most flexible and cost-effective
- Good for: Getting started, lower bandwidth needs

**Direct Peering**
- Router in same datacenter as Google point of presence
- 100+ Google PoPs worldwide
- Work with Carrier Peering partners if needed
- Good for: Direct traffic exchange with Google

**Dedicated Interconnect**
- One or more direct, private connections
- Can be backed up by VPN for reliability
- 10 or 100 Gbps per connection
- Good for: High bandwidth requirements, sensitive data

**Partner Interconnect**
- Connect through supported service provider
- For locations without Dedicated Interconnect access
- For workloads not needing full 10 Gbps
- Good for: Smaller bandwidth needs, remote locations

---

## Concluding Thoughts

Google Cloud's compute and networking services provide a powerful, flexible foundation for building applications. The combination of global VPCs, intelligent load balancing, autoscaling, and multiple connectivity options means you can architect solutions that are both performant and cost-effective.

Part 3 will explore storage options in Google Cloud, covering everything from object storage to databases.

---

*If you found this helpful, please share it with others learning GCP!*