# Research website

The static site in this directory is the public explainer for **How Much Attention Do You Actually Need?**

It is designed as an editorial research note rather than a portfolio dashboard. The page separates:
- the research question and controlled design;
- measured one-seed CPU pilots;
- the engineering evidence;
- the still-pending GPU sweep.

## Local preview

From the repository root:

```bash
python -m http.server 8000
```

Then open:

`http://localhost:8000/site/`

The two pilot figures are referenced from `figures/` in the repository. The GitHub Pages workflow copies them into the deployed site.

## Publishing

The workflow in `.github/workflows/pages.yml` deploys the site on pushes to `main` or the current project branch and can also be run manually.

In GitHub:
1. Settings → Pages
2. Source → **GitHub Actions**
3. Run the **site** workflow if it does not start automatically.

Expected project URL:

`https://mtk1606.github.io/How-Much-Attention-Do-You-Actually-Need/`

## Accuracy

`site/ACCURACY_AUDIT.md` maps every public quantitative claim to repository evidence. Pilot claims should remain labelled as pilots until the multi-seed GPU work exists.
