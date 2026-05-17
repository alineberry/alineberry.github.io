# adamlineberry.io

Personal site of Adam Lineberry — notes and writing on machine learning and
data science.

Built with [Astro](https://astro.build/), deployed to GitHub Pages via Actions.

## Local development

```sh
npm install
npm run dev
```

Then open <http://localhost:4321>.

To preview a production build:

```sh
npm run build
npm run preview
```

## Writing

- Posts live in `src/content/posts/` as Markdown (or MDX).
- Each post needs frontmatter:
  ```yaml
  ---
  title: "Post title"
  date: 2024-01-15
  description: "Short summary used for SEO and the post list."
  tags: ["machine learning", "deep learning"]
  permalink: /some/url/   # required; controls the URL exactly
  isNote: false            # set true to land it on /notes/ instead of homepage
  math: true               # enable KaTeX rendering for $...$ and $$...$$
  ---
  ```
- Images go in `public/images/` and are referenced as `/images/path.png`.

## Pages

- `src/pages/index.astro` — homepage (essays only)
- `src/pages/notes/index.astro` — notes archive
- `src/pages/about.astro` — about page
- `src/pages/[...slug].astro` — catch-all for post routes (uses `permalink` from frontmatter)

## Deploy

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push to `master`. Pages source must be set to "GitHub Actions" in the repo settings.
