# Website Personalization Roadmap

This document outlines the steps to personalize your Hugo website with your own content.

## General Repository Updates

### Updating the main content and the personal info
- ✔️ The `config.yml` file and the `content/` are all updated.

### Updating the Main README.md
- ✔️ The `README.md` file in the root of the repository has been updated to reflect that this is a personal website and not a template, while also acknowledging the original template provider and noting customizations.

### GitHub Pages Deployment Setup
- ✔️ GitHub Pages has been enabled and configured to use GitHub Actions for deployment. This involved setting the source to "GitHub Actions" in the repository settings, allowing the `hugo.yml` workflow to build and deploy the site.


# Additions to be Made ...


## 1. Adding Courses

Your website is already structured to handle courses. Here's how to add your "Matrix Algebra" and "Neural Network" courses.

### Directory Structure

All course content resides in the `/content/courses/` directory. You will create a new subdirectory for each course, and inside that, a markdown file for each tutorial.

Example structure:

```
/content/courses/
├───_index.md
├───matrix-algebra/
│   ├───index.md         # Main page for the Matrix Algebra course
│   ├───tutorial-01.md
│   ├───tutorial-02.md
│   └───...
└───neural-networks/
    ├───index.md         # Main page for the Neural Network course
    ├───tutorial-01.md
    ├───tutorial-02.md
    └───...
```

### Steps:

1.  **Create Course Directories:**
    *   Create a new folder: `/home/sam/Data/code/my-hugo-website/content/courses/matrix-algebra`
    *   Create another new folder: `/home/sam/Data/code/my-hugo-website/content/courses/neural-networks`

2.  **Create Course Content:**
    *   For each tutorial, create a new `.md` file inside the respective course directory (e.g., `tutorial-01.md`).
    *   Use the `course` archetype to generate new course pages with the correct front matter. You can do this with the command:
        ```bash
        hugo new courses/matrix-algebra/tutorial-01.md
        ```

3.  **Edit Tutorial Content:**
    *   Open the newly created markdown files and add your content. The front matter at the top of each file will look something like this. Fill in the details accordingly.

    ```yaml
    ---
    title: "Tutorial 1: Introduction to Vectors"
    date: 2025-06-29T21:00:00-07:00
    draft: false
    weight: 1 # Use weight to order your tutorials
    ---

    Your tutorial content goes here...
    ```

## 2. Creating a "News" Section

To add a section for news and tech trends, you'll need to create a new content section and add it to the main menu.

### Steps:

1.  **Create the Content Directory:**
    *   Create a new folder: `/home/sam/Data/code/my-hugo-website/content/news`

2.  **Add the "News" Section to the Menu:**
    *   Open your `config.yml` file.
    *   Add the following entry to the `menu.main` section:

    ```yaml
    menu:
        main:
            - name: News
              url: news/
              weight: 1 # Adjust weight to position it in the menu
            - name: Books
              url: books/
              weight: 5
            # ... other menu items
    ```
    *   Also, add `"news"` to the `MainSections` list under `params` in `config.yml` to ensure news posts appear on the main list pages.
    ```yaml
    params:
        MainSections: ["news", "books", "courses", "papers", "data"]
    ```

3.  **Create News Articles:**
    *   To create a new news article, you can use the `hugo new` command:
        ```bash
        hugo new news/my-first-tech-trend-post.md
        ```
    *   This will create a new markdown file in `/content/news/`. Edit this file to add your content.
