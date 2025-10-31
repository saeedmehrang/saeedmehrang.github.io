---
title: "Google Cloud Fundamentals: Learning Notes Summary (Part 1 of 4)"
date: 2025-10-28
draft: false
author: Saeed Mehrang
description: "A comprehensive guide to GCP fundamentals covering cloud service models, resource hierarchy, IAM, pricing, and essential concepts for getting started with Google Cloud Platform."
summary: "Essential GCP concepts in one place: service models (IaaS, PaaS, Serverless), resource hierarchy, Identity and Access Management, pricing structure, and ways to interact with Google Cloud."
tags: ["GCP", "Google Cloud", "Cloud Computing", "Cloud Fundamentals", "IAM", "Learning Notes"]
categories: ["Cloud", "Tutorial"]
series_order: 1
showToc: true
disableAnchoredHeadings: false
cover:
    image: /blogs/cloud-engineering-gcp/gcp-fundamentals/Google_Cloud_logo.svg
---


> **Course:** Google Cloud Fundamentals: Core Infrastructure | Google Cloud Skills Boost https://www.cloudskillsboost.google 
> **Focus:** Essential concepts and terminology for working with Google Cloud Platform

---

## Introduction

I recently started exploring Google Cloud Platform (GCP) in a structured way, moving beyond just-in-time learning. Following advice from a senior AI Engineer colleague, I'm progressing through fundamentals, then architecture, and finally AI Engineering—the area I need most.

This article captures the essential concepts in one place, making it easy to reference and refresh my understanding whenever needed.

## What's Covered

This first part explores foundational GCP concepts:

- Why companies migrate to the cloud
- Service models (IaaS, PaaS, Serverless)
- Infrastructure organization (Regions & Zones)
- Avoiding vendor lock-in
- Pricing structure
- Resource hierarchy
- Identity and Access Management (IAM)
- Ways to interact with GCP

*Part 2 will cover Virtual Machines and Networks, Part 3 will cover Storage, and finally Part 4 will cover Containers and Applications.*

---

## Why Migrate to Cloud?

Google's vision is clear: **every company will eventually become a data company**. Here's their thinking:

- Companies differentiate through technology
- That technology increasingly means software
- Great software depends on high-quality data
- Therefore, data capabilities define competitive advantage

Cloud platforms provide the infrastructure to become that data-driven company without building everything from scratch.

---

## Cloud Service Models: Understanding Your Options

Cloud computing evolved from physical data centers to virtualized resources. Here's how the different models compare:

| Service Model | What You Manage | What Provider Manages | Pricing Model | Best For |
|---------------|-----------------|----------------------|---------------|----------|
| **IaaS** (Infrastructure as a Service) | Applications, data, runtime, middleware | Servers, storage, networking | Pay for allocated resources | Maximum control over infrastructure |
| **PaaS** (Platform as a Service) | Applications and data only | Everything else including runtime | Pay for actual usage | Faster development without infrastructure management |
| **SaaS** (Software as a Service) | Just use the application | Everything | Subscription-based | End users consuming applications (Gmail, Google Drive) |

### Beyond Traditional Models

**Serverless Computing** represents the next evolution. Google's serverless offerings include:

- **Cloud Functions**: Event-driven code that runs without server management. You write code, Google handles everything else.
- **Cloud Run**: Deploy containerized applications in a fully managed environment.

**The Bottom Line**: These models differ in scalability, pricing, startup time, and deployment flexibility. Serverless and Cloud Run offer automatic scaling and flexibility, while IaaS and PaaS provide more infrastructure control.

---

## Infrastructure Layout: Regions and Zones

GCP's infrastructure spans **five major geographic areas**:
- North America
- South America  
- Europe
- Asia
- Australia

```
Organization Level: 37 Regions
    ↓
Region Level: 112 Zones (and growing)
    ↓
Your Application: Choose based on availability, durability, latency
```

**Why location matters**: Choosing where to deploy affects your application's availability, data durability, and user latency.

**Example**: Cloud Spanner can replicate your database across multiple zones within regions for even higher reliability.

---

## Avoiding Vendor Lock-In

Google understands that flexibility matters. If you need to move away from GCP, they provide tools to make it possible:

- **Kubernetes & Google Kubernetes Engine (GKE)**: Run microservices across different cloud providers
- **Google Cloud Operations Suite**: Monitor workloads across multiple clouds

This open approach means you're not trapped—you can mix and match services as your needs evolve.

---

## Pricing: Pay for What You Use

Google pioneered **per-second billing** for infrastructure services. Here's what makes their pricing competitive:

| Feature | Benefit |
|---------|---------|
| Per-second billing | No paying for unused minutes |
| Sustained use discounts | Automatic discounts for running VMs most of the month |
| Flexible billing | Available for Compute Engine, GKE, Dataproc, App Engine |

**Planning tip**: Use Google's online pricing calculator to estimate costs and set up budgets, alerts, reports, and quotas to stay in control.

---

## Resource Hierarchy: How GCP Organizes Everything

GCP uses a four-level hierarchy that determines how permissions and policies flow:

```
Level 4: Organization Node (top level)
    ↓
Level 3: Folders (group by department, team, environment)
    ↓
Level 2: Projects (separate billing and resource containers)
    ↓
Level 1: Resources (VMs, storage buckets, databases, etc.)
```

### Understanding Each Level

**Resources** (Level 1)  
The actual cloud services: virtual machines, Cloud Storage buckets, BigQuery tables, and more.

**Projects** (Level 2)  
Projects are the foundation for:
- Enabling and using Google Cloud services
- Managing APIs
- Handling billing
- Adding or removing team members

Each resource belongs to exactly one project. Projects can have different owners and are billed separately.

**Folders** (Level 3)  
Folders let you organize projects logically (by department, environment, or team) and apply policies at whatever level makes sense. Folders can contain projects, other folders, or both.

**Example structure**:
```
Organization: YourCompany
├── Folder: Engineering
│   ├── Folder: Development
│   └── Folder: Production
└── Folder: Marketing
    └── Project: Analytics
```

**Organization Node** (Level 4)  
The top-level container for everything in your GCP account. Special roles here include:
- **Organization Policy Administrator**: Controls policy changes
- **Project Creator**: Controls who can create projects (and spend money)

Note that folders can contain other folders (like Development and Production under Engineering) OR projects. The example above shows both patterns:
```
Engineering folder → contains subfolders (Development, Production)
Marketing folder → contains a project (Analytics)
```

This demonstrates the flexibility of the hierarchy. You could also have projects directly under Engineering if you wanted.

### Why This Matters

**Policies inherit downward**. Apply a policy at the folder level, and it automatically applies to all projects within that folder. This makes management efficient but requires careful planning.

---

## Identity and Access Management (IAM): Controlling Access

IAM answers three questions: **Who** can do **what** on **which resources**?

### The "Who" (Principals)

Principals can be:
- Google accounts (individual users)
- Google groups
- Service accounts (for applications)
- Cloud Identity domains

### The "What" (Roles)

Roles define permissions. GCP offers three types:

| Role Type | Description | Example | When to Use |
|-----------|-------------|---------|-------------|
| **Basic** | Broad permissions across entire project | Owner, Editor, Viewer, Billing Admin | Simple projects or getting started |
| **Predefined** | Service-specific permission bundles | Compute Instance Admin | Production environments with multiple team members |
| **Custom** | Precisely defined permissions you create | Instance Operator (start/stop only) | When predefined roles are too broad |

#### Basic Roles Breakdown

- **Owner**: Full control (resources, roles, permissions, billing)
- **Editor**: View and edit resources (but not permissions or billing)
- **Viewer**: View-only access
- **Billing Administrator**: View and manage billing only

#### When to Use Custom Roles

Custom roles work at the project or organization level (not folder level). Use them when you need precise control—for example, allowing users to start and stop VMs but not reconfigure them.

**Best Practice**: Follow the **least-privilege model**—give each person only the minimum permissions needed to do their job.

### Deny Rules

Deny rules override allow policies, providing an additional security layer. They prevent specific principals from using certain permissions regardless of their assigned roles. Like allow policies, deny rules inherit through the resource hierarchy.

---

## Service Accounts: Permissions for Applications

When applications or VMs need to access Google Cloud services without human intervention, use **service accounts**.

**Example use case**: A VM running an application that needs to store data in Cloud Storage. Create a service account with appropriate permissions, and assign it to the VM.

**Key points**:
- Service accounts are identities for applications, not people
- They can have IAM roles assigned to them
- Service accounts are also resources, so they can have their own IAM policies

---

## Accessing Google Cloud: Four Ways to Interact

| Method | Interface | Best For |
|--------|-----------|----------|
| **Cloud Console** | Web-based GUI | Visual management, learning, quick tasks |
| **Cloud SDK & Cloud Shell** | Command-line tools | Automation, scripting, developers |
| **APIs** | Programmatic access | Application integration, custom tools |
| **Google Cloud App** | Mobile application | Monitoring on the go, incident response |

### Cloud Console
Web-based interface for full control over services. Deploy, scale, and diagnose issues without leaving your browser.

### Cloud SDK & Cloud Shell

Command-line tools for Google Cloud:
- **gcloud**: Main CLI for GCP products and services
- **gcloud storage**: Access Cloud Storage
- **bq**: Interact with BigQuery

**Cloud Shell**: Don't want to install anything? Use Cloud Shell—a browser-accessible Debian VM with a persistent 5GB home directory and all tools pre-installed.

### APIs & Client Libraries

Google provides client libraries in popular languages:
- Java
- Python
- PHP
- C#
- Go
- Node.js
- Ruby
- C++

These libraries simplify calling Google Cloud services like Vertex AI, AutoML, and LLMs from your code.

### Google Cloud App

Mobile app features:
- SSH into VMs
- View billing and set up alerts
- Monitor key metrics (CPU, network, requests per second, errors)
- Manage incidents
- Create customizable dashboards

---

## Concluding Thoughts

GCP's architecture is clear and well-organized, making it easier to understand how components connect. Many engineers find GCP more intuitive than competing platforms, which means less time puzzling over configurations and more time building.

This first part covered the foundational concepts. Part 2 will cover virtual machines and networks.


---

*If you found this helpful, please share it with others learning GCP!*
