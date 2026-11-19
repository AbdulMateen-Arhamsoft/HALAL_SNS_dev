# Recommendation test page

A Streamlit page to try the recommendation engine (`recommendation/`) by hand: pick a candidate batch, move the
sliders, and see the scores, the diversity reordering and the formula for any single post.

## Run

From the repo root:

```
pip install -e ".[ui]"
streamlit run testui/recommendation/app.py
```

`.[ui]` installs Streamlit and pandas. The engine itself does not need them.

## Scenarios

Every `*.json` file in `tests/data/recommendation/` appears in the "Candidate scenario" box. A file has
`user_id`, `user_features` (`topic_weights`, `recent_creator_engagement`) and `candidates`. Add a file there
and it shows up on the next reload.

## Sidebar controls

The sliders build a modified copy of the settings (`get_settings().with_overrides(...)`) and pass it to the
engine. Nothing global changes, and the environment (`REC_*`) and `.env` values are the starting point.
"Reset to plan defaults" returns the sliders to those values.

| Control | Setting |
|---|---|
| Topic / Relationship / Freshness / Engagement / Exploration | `topic_weight`, `relationship_weight`, `freshness_weight`, `engagement_weight`, `exploration_weight` |
| Default half-life (hours) | `default_freshness_half_life_hours` |
| Followed half-life (hours) | `followed_freshness_half_life_hours` |
| Engagement smoothing k | `engagement_smoothing_k` |
| Engagement prior | `engagement_prior` |
| Save weight | `save_weight` |
| Max consecutive same creator | `max_consecutive_same_creator` |
| Max consecutive same topic | `max_consecutive_same_topic` |

The weights should sum to 1.0 (the page warns when they do not). `exploration_source` is not a slider.

## Files

- `app.py`: the page.
- `engine_runner.py`: runs the engine step by step for the page (score, sort, diversity, reasons) and explains one post. It does not import Streamlit.
