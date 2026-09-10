# Dataset build stats

Generated: 2026-09-10T06:34:49+00:00

## Inputs

- `/tmp/claude-0/-home-user-copywritingModel/f813ae63-f377-5d66-8e03-fd0bcbb1e21f/scratchpad/data/raw_yc.jsonl`
- `/tmp/claude-0/-home-user-copywritingModel/f813ae63-f377-5d66-8e03-fd0bcbb1e21f/scratchpad/data/raw_gallery.jsonl`

Raw records read per source:

| source | records read |
|---|---|
| gallery | 992 |
| yc | 4817 |

## Counts per split x source

| split | gallery | yc | total |
|---|---|---|---|
| train | 272 | 2094 | 2366 |
| val | 14 | 118 | 132 |
| test | 19 | 113 | 132 |
| **total** | 305 | 2325 | 2630 |

## Drop reasons

| reason | count |
|---|---|
| no_h1 | 1167 |
| too_few_h2 | 527 |
| too_few_button | 504 |
| non_english | 426 |
| too_few_p | 205 |
| dup_domain | 167 |
| h1_wordcount | 128 |
| forbidden_phrase:lorem ipsum | 17 |
| gallery_short_description | 15 |
| too_few_words | 11 |
| forbidden_phrase:404 | 6 |
| bad_title | 4 |
| forbidden_phrase:access denied | 1 |
| non_ascii_heavy | 1 |

## Target length distribution (kept examples, n=2630)

| metric | p10 | p50 | p90 |
|---|---|---|---|
| words per target | 207 | 395 | 418 |
| elements per target | 21 | 38 | 58 |

## Tag counts (kept examples, n=2630)

| tag | total count | avg per example |
|---|---|---|
| h1 | 2623 | 1.00 |
| h2 | 14585 | 5.55 |
| h3 | 15211 | 5.78 |
| h4 | 2379 | 0.90 |
| p | 50375 | 19.15 |
| button | 15565 | 5.92 |

## Example records

### Example 1 (train)

id: `yc-00959`  source: `yc`  url: https://www.sophyshealth.com/

Brief:
```
Product: Sophys
One-liner: AI Agents for Healthcare Intake
Industry: Healthcare
Tags: Artificial Intelligence, Analytics, Healthcare
```
Target (preview):
```
<h1>AI Agents for</h1>
<p>Custom Multimodal AI Agents for your Business</p>
<button>Get Started</button>
<h2>Wellness-Focused AI Solutions</h2>
<p>Purpose-built multimodal AI agents designed specifically for wellness platforms, mental health apps, and coaching services to enhance user engagement and deliver personalized support experiences.</p>
<h3>Empathetic User Interactions</h3>
<p>AI agents trained specifically for wellness communication, providing compassionate, supportive responses that maintain authentic connections and build trust with users.</p>
<h3>Multi-Platform Voice & Chat</h3>
<p>Seamlessly integrate multimodal agents into wellness apps, coaching platforms, and employee wellbeing programs with flexible APIs and customizable personalities.</p>
<h3>Behavioral Insights & Analytics</h3>
<p>Track user engagement patterns, wellness milestones, and behavioral trends to provide actionable insights that help improve app experiences and user outcomes.</p>
<h2>AI Companions for Every Wellness Journey</h2>
... (27 more lines)
```

### Example 2 (val)

id: `yc-02163`  source: `yc`  url: https://manicule.com

Brief:
```
Product: Manicule
One-liner: AgentRel — Devrel For Agents
Description: Manicule does AgentRel. We get companies discovered and used by AI agents. Think DevRel, rebuilt for a world where agents make all the decisions. We own developer documentation and technical content across social channels, end-to-end. We've grown 100% MoM over the last three months to $53K in monthly revenue, with customers like Supermemory, Greptile, Reducto, and Mastra. Getting discovered and used by agents is its own job now, and the market for it is broken.
Industry: B2B / Marketing
```
Target (preview):
```
<h1>Tomorrow's Users Aren't Developers. They're Agents.</h1>
<p>Manicule gets your product discovered and used by AI agents. Docs, GEO, and social content for how agents read. We provide the data on how agents use you, and the execution on top of it.</p>
<h2>See your docs the way agents see them.</h2>
<p>Drop a URL and we'll run the same audit our agents run on day one of an engagement: gaps, broken examples, dead ends, GEO surface area. Report in your inbox in about seven minutes.</p>
<button>Audit My Docs</button>
<p>~7 min · We'll email you the report</p>
<h2>Docs. Content. Distribution.</h2>
<p>We own three things for every customer: docs on our infra, content, and distribution across socials. Docs ship end to end in fourteen days. Every step has a clear owner.</p>
<h3>Audit</h3>
<p>Our agents crawl your docs, codebase, and support channels. They surface every gap, contradiction, and dead end. We build the diagnosis.</p>
<h3>Information Architecture</h3>
<p>The sidebar is the first thing developers and agents see. We map the journey from zero to “aha” and design a structure both can navigate without getting lost.</p>
... (16 more lines)
```

### Example 3 (test)

id: `yc-00477`  source: `yc`  url: http://vouch.us

Brief:
```
Product: Vouch
One-liner: Insurance and risk management tools for startups and growth-stage…
Description: Insurance... sounds slow, old-fashioned, and unexciting. Exactly. Insurance is broken, and it's failing fast-moving, innovative startups. Vouch is a new, technology-first insurance company backed with $185M in funding from world-class investors. Like Stripe for payments or Brex for credit cards, Vouch is creating the go-to business insurance for high-growth companies. We're doing this by making insurance fast, responsive, and focused on our high-growth and innovative customers.
Industry: Fintech / Insurance
Tags: Fintech, B2B
```
Target (preview):
```
<h1>The insurance broker for leaders building what's next.</h1>
<p>Industry expertise, a streamlined experience, and coverage that keeps pace with your growth.</p>
<button>Get Started</button>
<h2>Risk strategies for those who rewrite the rules</h2>
<h3>Technology</h3>
<p>Scale fast from AI breakthroughs to payment rails with insurance that keeps pace with innovation and regulation.</p>
<h3>Health & Life Sciences</h3>
<p>Navigate sensitive data and strict oversight with specialized coverage that moves you forward with confidence.</p>
<h3>Professional Services</h3>
<p>Keep your client work out of court with coverage built for accountants, consultants, creatives, and IT pros.</p>
<h3>Financial Services</h3>
<p>Withstand scrutiny from regulators, investors, and clients with coverage built for financial services.</p>
... (19 more lines)
```

