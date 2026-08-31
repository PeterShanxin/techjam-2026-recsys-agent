# Research-space audit (KuaiRand-Pure)

Inspected local files under `starter/kuairand/KuaiRand-Pure/data` plus `starter/kuairand/data.py`. Headers and counts come from those files, not filenames.

Official `data.load()` returns 7-tuples: `date, user_id, video_id, author_id, tab, duration_ms, long_view`. Encode fields: `user_id, video_id, author_id, tab, dur_bucket`. Official target: `long_view`. Research split: **valid**. **Test is sealed** for this sprint.

Official row counts: train **1,141,112** / valid **124,909** / test **170,588**.

Train date window in the loader is `20220408–20220421`. The train log file itself starts at **20220409** (no `20220408` rows).

## Files

| Raw file | Used by `data.load()` | Role |
| --- | --- | --- |
| `log_standard_4_08_to_4_21_pure.csv` | yes (partial columns) | Standard-exposure train log. 1,141,112 rows. Entire file is inside the train date window. |
| `log_standard_4_22_to_5_08_pure.csv` | yes (partial columns) | Standard-exposure valid+test log. 295,497 rows. Loader splits on date. |
| `log_random_4_22_to_5_08_pure.csv` | no | Random-exposure log. 1,186,059 rows. Dates overlap valid+test only. Extra unbiased check, not official fitness. |
| `video_features_basic_pure.csv` | yes (`author_id` only) | Video catalog. 7,583 rows / 7,583 videos. |
| `user_features_pure.csv` | no | User catalog. 27,285 rows / 27,285 users. |
| `video_features_statistic_pure.csv` | no | Unscoped video engagement counts. 7,583 rows. **No date window.** High leakage risk. |

## Interaction / context fields (standard + random logs)

Both standard logs and the random log share the same 19 columns.

| Field | Meaning | Train time | Inference time | Safe for valid research | Leakage risk | Cheap stats | Current access | Research uses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `user_id` | User key | yes | yes | yes | low | train 26,210 users; valid 22,377 | loader + encode | history, affinity, pairwise grouping |
| `video_id` | Video key | yes | yes | yes | low | train 7,538; valid+test file 6,618 | loader + encode | item stats, history match |
| `date` | `YYYYMMDD` | yes | yes (current row) | yes as recency *context*; not as a future label | low if used as current-row / train history time | train 13 days (`20220409–20220421`); valid 7 days; test 10 days | loader index 0 | recency, drift, history order |
| `tab` | Recommend tab | yes | yes | yes | low | 15 values on standard; 4 on random | loader + encode | context / residual |
| `duration_ms` | Video length (ms) | yes | yes | yes | low | train mean ~97.9k; max 1,177,720; zeros exist | loader; encode buckets it | duration / watch-bias |
| `long_view` | Official 0/1 target | train labels only | **no** | train labels only | **high** if valid/test labels enter features | train rate 0.337; valid rate 0.313 | loader index 6 | supervise train only |
| `hourmin` | Impression clock | yes | yes (current row) | yes as serving context | medium if treated as a label | present on all three logs | **not** on loader tuples | temporal / hour effects |
| `time_ms` | Impression timestamp | yes | yes (current row) | yes as serving context | medium | present on all three logs | **not** on loader | finer recency |
| `is_click` | Click on this impression | train aux | **no** (same-event label) | train aux / multi-task only | **high** on valid/test rows | train 528,845 pos | raw CSV / lab train aux | multi-task |
| `is_like` | Like on this impression | train aux | **no** | train aux only | **high** on valid/test rows | train 21,312 pos | raw CSV / lab train aux | multi-task, soft labels |
| `is_follow` | Follow on this impression | train aux | **no** | train aux only | **high** on valid/test rows | train 1,149 pos | raw CSV / lab train aux | rare aux |
| `is_comment` | Comment on this impression | train aux | **no** | train aux only | **high** on valid/test rows | train 2,930 pos | raw CSV / lab train aux | rare aux |
| `is_forward` | Forward/share on this impression | train aux | **no** | train aux only | **high** on valid/test rows | train 1,136 pos | raw CSV / lab train aux | rare aux |
| `is_hate` | Hate on this impression | train aux | **no** | train aux only | **high** on valid/test rows | train 480 pos | raw CSV / lab train aux | rare aux |
| `play_time_ms` | Watch time on this impression | train aux | **no** | train aux only | **high** on valid/test rows | present | raw CSV / lab train aux | watch-time / duration bias |
| `profile_stay_time` | Profile dwell | train aux | **no** | train aux only | **high** on valid/test rows | present | raw CSV | weak aux |
| `comment_stay_time` | Comment dwell | train aux | **no** | train aux only | **high** on valid/test rows | present | raw CSV | weak aux |
| `is_profile_enter` | Opened profile | train aux | **no** | train aux only | **high** on valid/test rows | present | raw CSV | weak aux |
| `is_rand` | Random-exposure flag | yes | yes if present | yes as context | low | `1` on random log | raw CSV | unbiased check |
| `author_id` | Uploader (joined) | yes | yes | yes | low | joined from video catalog; `UNK` if missing | loader (join only) | author affinity / history |
| interaction history | Prior user events | **train events only** | history from train, not from valid/test labels | yes if train-only | **high** if valid/test events are mixed in | ~44 train events / user mean | **not** indexed by loader | recency, sequence, affinity |
| exposure / popularity | Train counts / rates | train aggregates only | lookup OK | yes if train-only | **high** if valid/test rows update counts | compute from train | **not** precomputed | popularity, target encoding |

`(user_id, video_id)` is **not** unique on eval splits. Official `row_id` is file order after date filter. Do not key features on that pair alone.

## User catalog (`user_features_pure.csv`)

Static per `user_id`. Not read by `data.load()`. Available at train and serving as profile attributes. Safe as **features**, not as labels.

| Field | Meaning | Cardinality (cheap) | Leakage | Access | Uses |
| --- | --- | --- | --- | --- | --- |
| `user_id` | Join key | 27,285 | low | raw / lab | lookup |
| `user_active_degree` | Activity bucket | 9 | low | raw / lab | context |
| `is_lowactive_period` | Low-activity flag | **1** (constant here) | n/a | raw / lab | dead on this dump |
| `is_live_streamer` | Streamer flag | 2 | low | raw / lab | context |
| `is_video_author` | Author flag | 2 | low | raw / lab | context |
| `follow_user_num` / `_range` | Follow graph size | numeric / 8 buckets | low | raw / lab | organizer already found coarse user buckets weak vs `user_id` |
| `fans_user_num` / `_range` | Fan graph size | numeric / 9 buckets | low | raw / lab | same |
| `friend_user_num` / `_range` | Friend graph size | numeric / 7 buckets | low | raw / lab | same |
| `register_days` / `_range` | Account age | numeric / 8 buckets | low | raw / lab | same |
| `onehot_feat0`–`onehot_feat17` | Opaque user one-hots | 2–50+ | unknown semantics; treat as profile | raw / lab | residual / cross only (user-constant terms do not change within-user order) |

## Video catalog (`video_features_basic_pure.csv`)

Static per `video_id` except `visible_status` is constant here.

| Field | Meaning | Cardinality (cheap) | Train / infer | Safe | Leakage | Access | Uses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `video_id` | Join key | 7,583 | both | yes | low | raw / lab | lookup |
| `author_id` | Uploader | high | both | yes | low | loader join + catalog | affinity |
| `video_type` | Type | 3 | both | yes | low | raw / lab | context. Organizer: extra static fields did not help FM. |
| `upload_dt` | Upload date | 3 distinct in first 50 | both | yes | low–medium | raw / lab | item age vs impression date |
| `upload_type` | Upload path | 14 | both | yes | low | raw / lab | context |
| `visible_status` | Visibility | **1** | both | n/a | n/a | raw / lab | dead on this dump |
| `video_duration` | Catalog duration | high | both | yes | low | raw / lab; log also has `duration_ms` | duration bias |
| `server_width` / `server_height` | Resolution | high | both | yes | low | raw / lab | weak context |
| `music_id` / `music_type` | Music | high / 6 | both | yes | low | raw / lab | context; organizer static add was noise |
| `tag` | Video tag | high | both | yes | low | raw / lab | context |

## Video statistics (`video_features_statistic_pure.csv`) — unsafe default

52 columns: `show_cnt`, `play_cnt`, `like_cnt`, `complete_play_cnt`, `long_time_play_*`, `share_*`, `collect_*`, and other global counters. **No date, no split, no user.** These look like full-dump aggregates. They may include valid/test-period exposure.

| Use | Verdict |
| --- | --- |
| Default train popularity / target encoding | **Do not.** Compute from train interactions. |
| Explicit unscoped catalog probe | Allowed only if the candidate treats it as leaky / optional and does not hide that fact. |
| Official fitness | Never. |

## What was already practical

- Pointwise FM on the 5 encode fields
- FM bagging / more seeds
- Intra-seed checkpoint SWA + probability mean
- Raw-CSV reads if the candidate writes `csv.DictReader` / `csv.reader` itself

## What was missing (capability gap)

- No train-only history index
- No train-only popularity / affinity / target-rate helpers
- No pairwise sample builder
- No recency utilities
- No safe user/video catalog lookup
- No explicit train-vs-inference provenance
- ResearchState did not describe a lab, only the 7-tuple loader
- Starting priors stopped at FM + 3-seed ensemble (frozen SWA7 winner was not a Generation-0 prior)

## Practical families after this audit (agent still chooses)

These are now *possible* with NumPy + lab primitives. They are not a fixed experiment list.

1. History / recency / temporal
2. Pairwise ranking (BPR-style samples from train)
3. Listwise / within-user ranking on train exposures
4. Context / residual using raw user/video/author/tab/duration
5. Train-derived affinity / target encoding
6. Duration / watch-bias using train `play_time_ms` + `duration_ms`
7. Stronger interaction models that stay on NumPy (not another FM seed bag)

## Leakage contract

- Train-derived facts: **train split only**
- Inference-visible: current-row `user_id`, `video_id`, `author_id`, `tab`, `duration_ms`, `date`, plus catalog attributes and train history
- Valid `long_view` and valid/test aux actions: **not features**
- Test labels: **sealed**
- Random log: optional diagnostic, not official elite fitness
