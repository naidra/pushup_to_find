# Pushup To Find

React/Vite app prepared for GitHub Pages at:

`https://naidra.github.io/pushup_to_find/`

## Local Development

```sh
npm install
npm run dev
```

## GitHub Pages Build

```sh
npm run build:pages
```

This writes the static site to `dist/`, creates `404.html` for SPA routing, and adds `.nojekyll`.

## Deployment

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-pages.yml`.

In GitHub, set:

- Repository Settings -> Pages -> Build and deployment -> Source: `GitHub Actions`

Then push to `main`. The workflow will build and deploy the site automatically.
