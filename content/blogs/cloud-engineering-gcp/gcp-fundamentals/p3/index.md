---
title: "Google Cloud Fundamentals: Learning Notes Summary (Part 3 of 4)"
date: 2025-10-29
draft: false
author: Saeed Mehrang
description: "A comprehensive guide to GCP storage services: Cloud Storage, Cloud SQL, Cloud Spanner, Firestore, and Cloud Bigtable. Learn when to use each service with practical comparisons and decision frameworks."
summary: "Explore GCP's five core storage offerings: object storage with Cloud Storage, managed relational databases with Cloud SQL and Spanner, and NoSQL solutions with Firestore and Bigtable. Includes pricing, use cases, and a decision framework."
tags: ["GCP", "Google Cloud", "Cloud Storage", "Cloud SQL", "Cloud Spanner", "Firestore", "Bigtable", "NoSQL", "Database"]
categories: ["Cloud", "Tutorial"]
series_order: 3
showToc: true
disableAnchoredHeadings: false
cover:
    image: /blogs/cloud-engineering-gcp/gcp-fundamentals/Google_Cloud_logo.svg
---

> **Course:** Google Cloud Fundamentals: Core Infrastructure | Google Cloud Skills Boost https://www.cloudskillsboost.google 
> **Focus:** Storage in the Cloud

---

## Introduction

This is the third part of my systematic GCP fundamentals learning journey. While this article stands alone, you can read [Part 1](../p1) and [Part 2](../p2) for foundational concepts and networking details.

Part 3 explores Google Cloud's storage services—from object storage to managed databases. Every application needs to store data, and choosing the right storage solution is critical for performance, cost, and scalability.

## What's Covered

This article examines GCP's five core storage offerings:

- Cloud Storage (object storage)
- Cloud SQL (managed relational databases)
- Cloud Spanner (horizontally scalable SQL)
- Firestore (NoSQL for mobile and web)
- Cloud Bigtable (NoSQL for big data)

---

## Storage Types: Understanding Your Options

Google Cloud supports multiple storage paradigms to match different application needs:

| Storage Type | Characteristics | Example Use Cases |
|--------------|-----------------|-------------------|
| **Structured** | Organized in tables with schemas | Customer records, financial transactions |
| **Unstructured** | No predefined format | Images, videos, documents |
| **Transactional** | ACID compliance required | Order processing, banking operations |
| **Relational** | Connected data with relationships | User profiles with orders and addresses |

Most applications use multiple storage services to handle different data types and access patterns.

---

## Cloud Storage: Object Storage at Scale

Cloud Storage is Google's object storage service for binary large objects (BLOBs). Unlike file systems with folders or block storage with disk chunks, object storage manages data as discrete objects.

### How Object Storage Works

Each object contains:
- **Binary data**: The actual file content
- **Metadata**: Information like creation date, author, resource type, permissions
- **Unique identifier**: A globally unique URL

This URL-based approach makes object storage work seamlessly with web technologies.

### Common Use Cases

- Video and photo hosting
- Backup and archival storage
- Serving online content
- Storing intermediate processing results
- Data lake storage

### Organizing with Buckets

Cloud Storage files are organized into **buckets**. Each bucket requires:
- **Globally unique name**: No two buckets worldwide can share a name
- **Geographic location**: Choose based on user proximity to minimize latency

### Immutability and Versioning

**Important concept**: Cloud Storage objects are **immutable**—you don't edit them. Instead, each change creates a new version.

**Versioning options**:
- **Enabled**: Keep detailed history of all modifications, overwrites, and deletes
- **Disabled** (default): New versions always overwrite older versions

### Security and Access Control

Control access using:
- **IAM roles**: For broad, project-level permissions
- **Access Control Lists (ACLs)**: For fine-grained, object-level control

Follow the principle of least privilege: grant users access only to resources they need.

### Lifecycle Management

Storing large amounts of data gets expensive. Lifecycle policies automatically manage your data to reduce costs.

---

## Storage Classes: Matching Cost to Access Patterns

Choose the right storage class based on how frequently you access your data:

| Storage Class | Access Pattern | Use Case | Key Features |
|---------------|----------------|----------|--------------|
| **Standard** | Frequent access ("hot" data) | Active data, frequently accessed content | Best for data used daily/weekly |
| **Nearline** | ~Once per month | Data backups, multimedia archives | Low-cost for infrequent access |
| **Coldline** | ~Once per 90 days | Long-term backups, compliance archives | Lower cost, 90-day minimum |
| **Archive** | <Once per year | Disaster recovery, regulatory retention | Lowest cost, 365-day minimum, higher access costs |

### Auto-Class: Intelligent Storage Management

Auto-class automatically transitions objects between storage classes based on actual access patterns:

```
Frequently Accessed → Standard Storage (optimized performance)
    ↓
No Access for 30 Days → Nearline Storage (reduced cost)
    ↓
No Access for 90 Days → Coldline Storage (lower cost)
    ↓
No Access for 365 Days → Archive Storage (lowest cost)
```

When data is accessed again, auto-class moves it back to Standard storage for optimal performance.

---

## Data Transfer: Getting Data into Cloud Storage

### Transfer Methods Compared

| Method | Best For | Capacity |
|--------|----------|----------|
| **Storage Transfer Service** | Online data from another cloud, region, or HTTPS endpoint | Large-scale, scheduled transfers |
| **Transfer Appliance** | Offline data transfer from on-premises | Up to 1 petabyte per appliance |
| **Integration with GCP services** | BigQuery, Cloud SQL imports/exports | Varies by service |

### Storage Transfer Service

Schedule and manage batch transfers:
- From another cloud provider (AWS, Azure)
- From a different Cloud Storage region
- From any HTTPS endpoint

Ideal for migrating large datasets or setting up recurring transfers.

### Transfer Appliance

For massive on-premises datasets:
1. Lease a rackable storage server from Google
2. Connect it to your network
3. Load your data
4. Ship it to Google's upload facility
5. Google uploads data to Cloud Storage

Perfect when network transfer isn't practical due to bandwidth constraints or data volume.

---

## Encryption and Security

**Server-side encryption**: Always enabled at no additional charge. Data is encrypted before being written to disk.

**Data in transit**: Encrypted by default using HTTPS/TLS when traveling between your device and Google.

**No provisioning required**: Unlike traditional storage, you don't need to allocate capacity in advance. Cloud Storage scales automatically.

---

## Cloud SQL: Managed Relational Databases

Cloud SQL provides fully managed relational databases without software installation or maintenance burden.

### Supported Database Engines

- MySQL
- PostgreSQL
- SQL Server

### Specifications and Scaling

| Resource | Maximum Capacity |
|----------|------------------|
| Processor Cores | 128 cores |
| RAM | 864 GB |
| Storage | 64 TB |

### Built-In Features

**Managed Backups**
- Securely stored and accessible for restore
- Seven backups included in instance cost
- Automated scheduling

**Security**
- Encryption on Google's internal networks
- Encryption for database tables, temporary files, and backups
- Network firewall controlling database access

### Integration with Google Cloud

Cloud SQL instances are accessible by:

**App Engine**: Use standard database drivers
- Connector/J for Java applications
- MySQLdb for Python applications

**Compute Engine**: Authorize VM instances to access Cloud SQL
- Place instances in the same zone for optimal performance

**External Applications**: Connect using standard MySQL drivers
- SQL Workbench
- Toad
- Other third-party tools

---

## Cloud Spanner: Globally Distributed SQL

Cloud Spanner is Google's horizontally scalable, strongly consistent relational database service. It powers Google's $80 billion business and mission-critical applications.

### When to Choose Cloud Spanner

Cloud Spanner is ideal when you need:

| Requirement | Why Spanner |
|-------------|-------------|
| **SQL relational database** | Full SQL support with joins and secondary indexes |
| **Global scale** | Horizontal scaling across regions |
| **High availability** | Built-in, no configuration needed |
| **Strong consistency** | ACID transactions globally |
| **High throughput** | Tens of thousands of reads/writes per second or more |

### Cloud SQL vs. Cloud Spanner

```
Choose Cloud SQL when:
- Single region is sufficient
- Up to 64 TB is enough
- Read replicas provide sufficient scaling

Choose Cloud Spanner when:
- Global distribution required
- Petabyte-scale capacity needed
- Horizontal write scaling essential
- Strong consistency across regions mandatory
```

---

## Firestore: NoSQL for Mobile and Web

Firestore is a flexible, horizontally scalable NoSQL database designed for mobile, web, and server development.

### Data Model

**Documents and Collections**: Data is organized hierarchically.

```
Collection: users
├── Document: user123
│   ├── firstname: "John"
│   ├── lastname: "Doe"
│   └── Subcollection: orders
│       └── Document: order456
└── Document: user789
    ├── firstname: "Jane"
    └── lastname: "Smith"
```

### Key Features

**Flexible Queries**
- Retrieve individual documents
- Query collections with multiple chained filters
- Combine filtering and sorting
- Indexed by default for consistent performance

**Query performance is proportional to result set size, not dataset size.**

**Real-Time Synchronization**
- Data updates automatically sync across all connected devices
- Works with mobile apps, web apps, and servers

**Offline Support**
- Apps can write, read, listen to, and query data while offline
- Caches actively used data locally
- Syncs changes when device reconnects

### Infrastructure Benefits

Leverages Google Cloud's powerful infrastructure:
- Automatic multi-region data replication
- Strong consistency guarantees
- Atomic batch operations
- Real transaction support

### Pricing Model

You're charged per operation:
- Each document read
- Each document write
- Each document delete
- Each query (counts as one document read, regardless of results)

**Free Daily Quota**:
- 50,000 document reads
- 20,000 document writes
- 20,000 document deletes
- 1 GB stored data
- 10 GiB network egress per month (between US regions)

---

## Cloud Bigtable: NoSQL for Big Data

Cloud Bigtable is Google's NoSQL big data database service, designed for massive workloads with consistent low latency and high throughput. It's the same database powering Google Search, Analytics, Maps, and Gmail.

### When to Choose Cloud Bigtable

Select Bigtable when your requirements include:

| Criterion | Threshold/Characteristic |
|-----------|-------------------------|
| **Data Volume** | More than 1 TB of data |
| **Data Velocity** | Fast, high throughput, or rapidly changing |
| **Data Type** | NoSQL without strong relational semantics |
| **Data Structure** | Time-series or natural semantic ordering |
| **Processing** | Asynchronous batch or synchronous real-time |
| **Machine Learning** | Running ML algorithms on the data |

### Perfect Use Cases

- **Internet of Things (IoT)**: Sensor data, device telemetry
- **User Analytics**: Clickstreams, user behavior tracking
- **Financial Data**: Trading data, market analysis
- **Time-Series Data**: Monitoring, logs, metrics

### Reading and Writing Data

**Application Data Service Layer**

Serve data to applications through:
- Managed VMs
- HBase REST Server
- Java Server using HBase client

Used for: Applications, dashboards, data services

**Streaming and Batch Processing**

| Processing Type | Frameworks | Use Case |
|-----------------|-----------|----------|
| **Streaming** | Dataflow Streaming, Spark Streaming, Storm | Real-time data flows |
| **Batch** | Hadoop MapReduce, Dataflow, Spark | Bulk operations, historical analysis |

Processed data is often written back to Bigtable or downstream databases.

---

## Comparing Storage Options: Choosing the Right Service

Here's a comprehensive comparison to help you choose the right storage service:

| Service | Primary Use Case | Scaling/Capacity | Key Constraints | Best For |
|---------|------------------|------------------|-----------------|----------|
| **Cloud Storage** | Immutable blobs (images, videos, files) | Petabytes; max object: 5 TB | Objects >10 MB recommended | Large media files, backups, data lakes |
| **Cloud SQL** | Full SQL, OLTP | Up to 64 TB | Machine type dependent | Web apps, traditional databases, existing applications |
| **Cloud Spanner** | Full SQL, OLTP with horizontal scaling | Petabytes | - | Global apps requiring SQL and massive scale |
| **Firestore** | Mobile/web app data with sync | Terabytes; max entity: 1 MB | - | Real-time apps, mobile backends, offline-capable apps |
| **Cloud Bigtable** | Large structured objects, heavy read/write | Petabytes; max cell: 10 MB, row: 100 MB | No SQL queries or multi-row transactions | Analytics, IoT, time-series, financial data |

### Decision Framework

**Start with these questions:**

1. **Is your data structured or unstructured?**
   - Unstructured (media, files) → **Cloud Storage**
   - Structured → Continue to next question

2. **Do you need SQL?**
   - Yes → Continue to next question
   - No → Continue to NoSQL options

3. **SQL: Do you need global scale beyond 64 TB?**
   - No → **Cloud SQL**
   - Yes → **Cloud Spanner**

4. **NoSQL: What's your primary use case?**
   - Mobile/web with real-time sync → **Firestore**
   - Big data analytics, IoT, high throughput → **Cloud Bigtable**

### What About BigQuery?

**BigQuery** sits at the edge between storage and processing. While it stores data, its primary purpose is enabling big data analysis and interactive querying. Think of it as an analytical engine rather than a pure storage service.

---

## Concluding Thoughts

Google Cloud's storage services provide solutions for every data type and access pattern. Understanding the strengths of each service helps you architect efficient, cost-effective applications:

- Use **Cloud Storage** for objects and files
- Choose **Cloud SQL** for traditional relational workloads
- Select **Cloud Spanner** when you need SQL at global scale
- Pick **Firestore** for mobile and web apps requiring real-time sync
- Opt for **Cloud Bigtable** for big data analytics and IoT workloads

Part 4 will explore containers and application deployment on Google Cloud.

---

*If you found this helpful, please share it with others learning GCP!*