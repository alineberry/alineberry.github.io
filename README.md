# adamlineberry.io

Personal site of Adam Lineberry — notes and writing on machine learning and
data science.

Built with [Jekyll](https://jekyllrb.com/) and the
[Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) theme,
hosted on GitHub Pages.

## Local development

```sh
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

## Writing

- Posts live in `_posts/` with the filename convention `YYYY-MM-DD-slug.md`.
- Pages live in `_pages/`.
- Images live in `images/` (per-post subfolder for anything beyond a hero).

## Configuration

Site-wide settings are in `_config.yml`. The theme is pulled in via
`remote_theme`, so upgrading is a one-line bump.
