---
title: "Sample Blog Post with Bundled Resources"
date: 2024-09-15
tags: ["tutorial", "machine learning", "examples"]
author: "Saeed Mehrang"
description: "A sample blog post demonstrating how to include bundled images and PDFs."
summary: "This sample post shows how page bundles work in Hugo, allowing you to include images and PDFs alongside your markdown content."
cover:
    image: "sample-image.png"
    alt: "Sample Cover Image"
    relative: true
showToc: true
disableAnchoredHeadings: false

---

## Introduction

This is a sample blog post that demonstrates how to use **page bundles** in Hugo. With page bundles, you can include images, PDFs, and other resources in the same directory as your markdown file.

## Including Images

You can reference images that are in the same directory as this `index.md` file using relative paths:

![Sample Image](sample-image.png)

The image above is stored in the same directory as this blog post.

## Including PDFs

You can also link to PDF files stored in the page bundle:

### Lecture Materials

+ [Lecture Notes 1](lecture-notes-1.pdf)
+ [Lecture Notes 2](lecture-notes-2.pdf)
+ [Problem Set](problem-set.pdf)

### Research Papers

+ [Paper 1](research-paper-1.pdf) - This is a sample research paper
+ [Paper 2](research-paper-2.pdf) - Another sample paper with detailed analysis

## How Page Bundles Work

In Hugo, a **page bundle** is a directory that contains:

1. An `index.md` file (the main content)
2. Any resources (images, PDFs, data files) that you want to bundle with the page

For this blog post, the structure looks like:

```
content/blogs/sample-blog-with-resources/
├── index.md
├── sample-image.png
├── lecture-notes-1.pdf
├── lecture-notes-2.pdf
├── problem-set.pdf
├── research-paper-1.pdf
└── research-paper-2.pdf
```

## Benefits

+ **Organization**: All related files are kept together
+ **Portability**: Easy to move or copy entire blog posts
+ **Relative Links**: No need to worry about absolute paths
+ **Clean URLs**: Resources are accessible at clean URLs like `/blogs/sample-blog-with-resources/sample-image.png`

## Conclusion

This is exactly how the courses section works, and now the blogs section has the same capability!
