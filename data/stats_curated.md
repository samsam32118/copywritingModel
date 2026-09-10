# Dataset curation stats

## Counts per split and per source

| split | source | count |
|---|---|---|
| train | gallery | 132 |
| train | yc | 1179 |
| **train** | **total** | **1311** |
| val | gallery | 11 |
| val | yc | 89 |
| **val** | **total** | **100** |
| test | gallery | 14 |
| test | yc | 106 |
| **test** | **total** | **120** |
| **all** | **total** | **1531** |

## Curation and re-split summary

| input file | input records | passed rules 1-3 | drop rate |
|---|---|---|---|
| train.jsonl | 2366 | 1374 | 41.9% |
| val.jsonl | 132 | 78 | 40.9% |
| test.jsonl | 132 | 79 | 40.2% |

Re-split (seed=42): moved **41** curated train records into test and **22** into val (domain-disjoint from val/test), to reach the targets. Final: train=1311, val=100, test=120.

## Drop histogram (first failing rule per record)

| rule | train.jsonl | val.jsonl | test.jsonl | total | description |
|---|---|---|---|---|---|
| hero_not_in_first3 | 274 | 15 | 16 | 305 | Rule 2: h1 present but not within the first 3 elements |
| h1_count | 7 | 0 | 0 | 7 | Rule 3: h1 count != 1 |
| h1_words | 6 | 0 | 0 | 6 | Rule 3: h1 word count not in [2,16] |
| p_count | 150 | 9 | 2 | 161 | Rule 3: p count not in [4,35] |
| h2_min | 42 | 4 | 3 | 49 | Rule 3: h2 count < 2 |
| h2_ratio | 24 | 1 | 0 | 25 | Rule 3: h2 count / element count > 0.45 |
| max_run | 368 | 21 | 25 | 414 | Rule 3: longest consecutive same-tag run > 8 |
| total_words | 42 | 1 | 3 | 46 | Rule 3: total words not in [120,420] |
| no_h3h4_or_cta | 38 | 1 | 0 | 39 | Rule 3: no h3/h4 present and button count < 2 |
| line_too_long | 2 | 1 | 0 | 3 | Rule 3: a line has > 90 words |
| button_repeat | 1 | 0 | 0 | 1 | Rule 3: a button text (case-folded) repeated > 2 times |
| numeric_lines | 6 | 0 | 0 | 6 | Rule 3: >= 4 pure number/price/date lines |
| banned_phrase | 32 | 1 | 4 | 37 | Rule 3: a line contains "cookie" / "javascript" / "subscribe to our newsletter" (case-insensitive) |
| **total dropped** | | | | **1099** | |

## Curated train: words / elements per target (percentiles)

| metric | p10 | p50 | p90 |
|---|---|---|---|
| words | 213.0 | 391.0 | 418.0 |
| elements | 22.0 | 35.0 | 51.0 |

## Curated train: mean count per tag per record

| tag | mean count |
|---|---|
| h1 | 1.000 |
| h2 | 5.748 |
| h3 | 6.485 |
| h4 | 0.963 |
| p | 16.435 |
| button | 5.044 |

## Example curated train targets (ids ending in 5 and 7, among the first 20)

### id ending in '5'

id: `yc-00405`  name: Bottomless  url: https://bottomless.com

```
<h1>How it works</h1>
<button>Personalize my rotation</button>
<button>Shop coffee</button>
<h2>How do you like your coffee?</h2>
<h2>Build your own coffee rotation</h2>
<h2>Pricing</h2>
<button>Get started</button>
<h2>Roasted to order from partners across the U.S.</h2>
<h2>Customer reviews</h2>
<h2>Frequently asked questions</h2>
<p>Start by taking a quiz to find your perfect coffee rotation. We'll send you a Bottomless scale with your first order. Store your coffee on the scale, and it will reorder your next bag at the perfect time.</p>
<p>Our rotation engine is the most advanced ever built. You can specify almost any possible coffee configuration, and we pick the perfect coffee for you based on your location, preferences and ratings.</p>
<p>We have an online store featuring a variety of roasters. If you like consistency, you can pick a specific coffee to be reordered repeatedly. If you like variety, you can choose one of our rotations or create your own based on your preferences. Change the products linked to your account at any time.</p>
<p>Membership costs $7.99/month and includes the scale and unlimited shipping. Additionally, you'll get 20% off your first bag and a 10% discount off each subsequent order.</p>
<p>Signing up for a Bottomless membership directly offers you access to our full marketplace and a customized rotation of different brands for a monthly fee. Signing up with a brand gives you access to Subscribe by Usage with one brand at no additional cost.</p>
<p>Yes, you will get an SMS or email alert 8-12 hours before reorders are placed, giving you a chance to change your product, postpone, or cancel. You can also customize the "eagerness" of the algorithm by opting for "never running out," "just right shipments," or "prioritizing freshness."</p>
<p>Yes! The scale can easily be "zeroed" with almost any home coffee container or jar.</p>
<p>A ton of our customers do this. It works perfectly fine, just pour less than half a bag into the hopper at a time.</p>
<h2>Try Bottomless for 30 days</h2>
<button>Start trial</button>
```

### id ending in '7'

id: `yc-00917`  name: Scout  url: http://scouthealth.com

```
<h1>Faster answers. Better treatment.</h1>
<p>Portable molecular testing offering lab-grade performance in under 30 minutes.</p>
<button>Join Waitlist</button>
<p>1 This product is under development and performance claims are based on clinical trial data that have not been reviewed by the FDA, see footer for important information.</p>
<h2>Definitive results without the wait.</h2>
<p>Scout's patented molecular technology delivered results matching high-complexity PCR 99% of the time 1 , allowing clinicians and patients to make immediate care decisions.</p>
<button>Request more information</button>
<h2>Bridging the gap between diagnosis and care</h2>
<p>Traditional diagnostics force a trade-off between accuracy, speed, and cost. Scout's innovative technology eliminates that compromise.</p>
<h3>Lateral Flow Tests</h3>
<p>Centers for Disease Control and Prevention. (2024, April 25). SARS-CoV-2 viral shedding and rapid antigen test performance — Respiratory Virus Transmission Network, November 2022–May 2023. Morbidity and Mortality Weekly Report, 73(16), 365–371.</p>
<p>Fast but miss early infections. High false-negative rates lead to undertreated patients.</p>
<h3>Scout</h3>
<p>Molecular accuracy at a lower cost than any other CLIA-waived system</p>
<h3>Central Lab PCR</h3>
<p>Gold standard performance, but slow turnaround times delay treatments and increases patient loss to follow-up</p>
<h2>Backed by world-class research institutions.</h2>
<p>Scout has been awarded funding from the NIH, CARB-X, and Flu Lab, as well as support from institutions such as Johns Hopkins University and Emory University</p>
<p>" Scout's innovative testing platform allowed Medcor to support our Hollywood productions in remote environments, bypassing lab limitations and enabling beloved shows to continue safely during the pandemic. "</p>
<h2>A convenient solution for wherever care is needed</h2>
<p>Scout is ultra-portable and shelf stable, and can be deployed wherever care is needed (no installation or training required).</p>
<button>Clinic</button>
<button>Community</button>
<button>Home</button>
<p>Implement low-cost NAAT testing at your practice. Test patients and receive results during their visit, enabling decisive treatment and complete care. Billing may be available using applicable CPT codes, depending on payer policies.</p>
<h2>Interested in transforming near-patient care?</h2>
<p>Join leading clinics using Scout for faster, more accurate, and cost-effective patient care.</p>
```

