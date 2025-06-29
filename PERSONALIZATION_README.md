# Website Personalization Roadmap

This document outlines the steps to personalize your Hugo website with your own content.

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

## 3. Adding Work Experience and Education

You have a few options for adding your professional background.

### Option A: Update the Homepage Profile (Easiest)

Your homepage already has a profile section. You can quickly update this with your experience and education.

1.  **Edit `config.yml`:**
    *   Open the `config.yml` file.
    *   Find the `params.profileMode.subtitle` section.
    *   Replace the existing text with your own bio, including your work experience and education.

### Option B: Create a Detailed CV/Resume Page

For a more detailed presentation, you can create a dedicated page.

1.  **Create an "About" Page:**
    *   Create a new file: `/home/sam/Data/code/my-hugo-website/content/about.md`
    *   Add your content in Markdown format. You can structure it with headings for "Work Experience," "Education," "Skills," etc.
    *   Example front matter for `/content/about.md`:
    ```yaml
    ---
    title: "About Me"
    date: 2025-06-29
    draft: false
    ---

    ## Education
    *   **Ph.D. in Computer Science** - University of Example (2020)
    *   **B.S. in Mathematics** - Example College (2016)

    ## Work Experience
    *   **Senior AI Researcher** - Tech Corp (2020 - Present)
    *   ...
    ```

2.  **Add "About" to the Menu:**
    *   If you want a link to this new page in your main menu, add it to `config.yml` as you did for the "News" section.

### Option C: Link to your PDF CV

Your site already includes a `cv.pdf` file in the `static` folder and a link to it in the social icons. You can simply update this PDF file with your latest information.

*   **File Location:** `/home/sam/Data/code/my-hugo-website/static/cv.pdf`
*   Replace the existing file with your own CV. The "CV" link on your homepage will automatically point to the new file.

## 4. Personalizing Site-Wide Information

Several key details of your website are controlled by the `config.yml` file. Here are the most important ones to change:

*   **`title`**: The main title of your website (e.g., "Sam's Portfolio & Blog").
*   **`author`**: Your name, which appears in metadata.
*   **`description`**: A short description of your site for search engines.
*   **`params.profileMode`**: This controls the main homepage profile.
    *   **`title`**: Your name or professional title.
    *   **`subtitle`**: Your bio, which can include your work experience and education.
    *   **`imageUrl`**: The path to your profile picture (e.g., `"picture.jpeg"`). Make sure the image is in the `/static/` directory.
*   **`params.socialIcons`**: This is a list of your social media and other links.
    *   You can edit or remove the existing entries.
    *   To add a new one, follow the existing format (e.g., `- name: LinkedIn
      url: https://www.linkedin.com/in/your-profile`).
*   **`menu.main`**: This controls the main navigation menu at the top of the page. You can add, remove, or reorder items here.
