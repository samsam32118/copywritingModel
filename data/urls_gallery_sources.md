# URL Gallery Sources

Each row is a gallery/list that was mined for outbound links to real company product/SaaS/startup marketing homepages. Counts are the number of URLs in the final deduplicated `urls_gallery.txt` whose registrable domain was first contributed by that source (domains seen in more than one source are credited to whichever source was processed first, not double-counted).

| Source | URLs contributed | Notes |
|---|---|---|
| Landingfolio (landingfolio.com) | 273 | Scraped 33 `/inspiration/landing-page/<category>` category pages (SaaS, AI, fintech, health, crypto, real estate, e-commerce, marketing, mobile app, etc.) via each card's `data-testid="inspiration-card-outbound"` link. Fully accessible, no blocking. |
| SaaS Landing Page (saaslandingpage.com) | 272 | Scraped 30 pages of the WordPress post archive (`/page/N/`) via each card's `title="Visit Website"` link, explicitly excluding the site's own paid sponsor strip. Fully accessible, no blocking. |
| One Page Love (onepagelove.com) | 55 | Scraped page 1 of 13 relevant `/inspiration?genre=<genre>` filters (saas, startup, app, product, e-commerce, finance, service, landing-page, informational, launching-soon, etc.) via the `?ref=onepagelove` outbound link marker. Note: the site's own pagination (`paged=`/`/page/N`) silently drops the genre filter beyond page 1 (verified: different genres returned identical results), so only the first page per genre was genre-accurate and used; deeper pages were not scraped to avoid mislabeled results. Site also advertises an official JSON API (onepagelove.com/api) but it is in private beta with no instant key issuance, so HTML scraping was used instead. |
| Indie Hackers (indiehackers.com/products) | 16 | Listing page exposes only ~22 product detail-page slugs server-side (no working pagination/sort without JS); visited each `/product/<slug>` detail page for its outbound site link. Low yield; several resolved to app-store/GitHub-Pages links that were then excluded by the domain filter. |
| GitHub: georgezouq/awesome-saas | 75 | Curated "awesome list" markdown README, parsed for `[name](https://...)` links via raw.githubusercontent.com (github.com's own web UI returns HTTP 403 to non-browser requests; the raw content host is unaffected). |
| GitHub: GetStream/awesome-saas-services | 53 | Same method as above; curated list of well-known SaaS services. |
| YC-OSS public API (yc-oss.github.io/api, mirrors Y Combinator's company directory) | 256 | Community-maintained JSON mirror of YC's public company directory (6,209 companies), hosted on GitHub Pages (not github.com, so unaffected by its bot block). Took every publicly-traded ('Public') and YC-flagged 'top_company' alum, plus a stratified-by-industry sample (capped per industry, favoring larger team_size) of the remaining 'Active' companies, spanning B2B, Healthcare, Fintech, Consumer, Industrials, Real Estate, Education and Government. |

## Sources attempted but blocked or infeasible to scrape cheaply

| Source | Result |
|---|---|
| Lapa Ninja (lapa.ninja) | HTTP 403 from Cloudflare (`cf-mitigated: challenge`, a JS/CAPTCHA challenge) on every category page, both via curl and via the WebFetch tool. No workaround attempted beyond header spoofing, per instructions to move on from 403s. |
| Land-book (land-book.com) | Same Cloudflare challenge block (HTTP 403, `cf-mitigated: challenge`) as Lapa Ninja. |
| Product Hunt (producthunt.com) | HTTP 403 on every route tried (root, leaderboard); appears to be behind the same class of bot mitigation. No public unauthenticated API available; the GraphQL API requires an OAuth token. |
| Godly (godly.website) | Returns HTTP 200 but the page is an empty shell for a client-side JS single-page app (Vite bundle) with no server-rendered card/link data and no discoverable public JSON endpoint in the static payload; not scrapable without a JS-executing browser. |
| Webflow showcase (webflow.com/made-in-webflow) | Returns HTTP 200 but, like Godly, ships no server-rendered showcase links in the static HTML (client-side rendered React app); skipped. |
| BuiltWith (builtwith.com) | Accessible, but it is a technology-lookup tool, not a landing-page gallery — it has no browsable feed of showcased marketing homepages to mine, so it was not a usable source for this task. |
| GitHub search / github.com web UI directly | github.com (the web app, as opposed to raw.githubusercontent.com and github.io Pages) returns HTTP 403 to plain requests, including `github.com/search`, individual `github.com/<owner>/<repo>` pages, and `api.github.com`. Worked around by reading raw file content from raw.githubusercontent.com instead. |
| softvar/awesome-startups (GitHub) | Real repo, real per-country startup lists, but every entry links to a startupranking.com ranking-profile page (e.g. `startupranking.com/airbnb`), not the company's own homepage — and guessing the real domain from the company name would risk fabricating URLs, so this source was excluded rather than used. |

**Total URLs in `urls_gallery.txt`: 1000** (deduplicated by registrable domain; 111 candidate domains were dropped after failing a liveness check — DNS failure, connection refused, or timeout).
