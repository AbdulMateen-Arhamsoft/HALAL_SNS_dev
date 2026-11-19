from recommendation.config import RecommendationSettings, get_settings
from recommendation.diversity import apply_diversity, has_same_topic, violates_diversity

CFG = RecommendationSettings(_env_file=None)


def post(pid: str, creator: str, topics: list[str]) -> dict:
    return {"post_id": pid, "creator_id": creator, "topics": topics}


def ids(posts: list[dict]) -> list[str]:
    return [p["post_id"] for p in posts]


def test_has_same_topic() -> None:
    assert has_same_topic(["a", "b"], ["b"])
    assert not has_same_topic(["a"], ["b"])
    assert not has_same_topic([], [])


def test_no_more_than_two_in_a_row_from_one_creator() -> None:
    posts = [post("1", "A", ["t1"]), post("2", "A", ["t2"]), post("3", "A", ["t3"]), post("4", "B", ["t4"])]
    assert ids(apply_diversity(posts, CFG)) == ["1", "2", "4", "3"]


def test_no_more_than_two_in_a_row_on_one_topic() -> None:
    posts = [post("1", "A", ["x"]), post("2", "B", ["x"]), post("3", "C", ["x"]), post("4", "D", ["y"])]
    assert ids(apply_diversity(posts, CFG)) == ["1", "2", "4", "3"]


def test_delayed_posts_are_not_dropped() -> None:
    posts = [post(str(i), "A", ["x"]) for i in range(3)] + [post("9", "B", ["y"])]
    out = apply_diversity(posts, CFG)
    assert sorted(ids(out)) == sorted(ids(posts))


def test_all_same_creator_keeps_every_post_in_order() -> None:
    posts = [post(str(i), "A", ["t"]) for i in range(6)]
    assert ids(apply_diversity(posts, CFG)) == ids(posts)


def test_empty_list() -> None:
    assert apply_diversity([], CFG) == []


def test_input_list_is_not_mutated() -> None:
    posts = [post("1", "A", []), post("2", "A", []), post("3", "A", []), post("4", "B", [])]
    before = list(posts)
    apply_diversity(posts, CFG)
    assert posts == before


def test_posts_without_topics_are_not_topic_limited() -> None:
    posts = [post(str(i), f"c{i}", []) for i in range(5)]
    assert ids(apply_diversity(posts, CFG)) == ids(posts)


def test_violates_diversity_needs_enough_history() -> None:
    assert not violates_diversity(post("2", "A", ["x"]), [post("1", "A", ["x"])], CFG)
    assert violates_diversity(post("3", "A", ["z"]), [post("1", "A", ["x"]), post("2", "A", ["y"])], CFG)


def test_config_override_limit_of_one() -> None:
    cfg = CFG.with_overrides(max_consecutive_same_creator=1)
    posts = [post("1", "A", ["t1"]), post("2", "A", ["t2"]), post("3", "B", ["t3"])]
    assert ids(apply_diversity(posts, cfg)) == ["1", "3", "2"]


def test_config_override_larger_limit_allows_longer_runs() -> None:
    cfg = CFG.with_overrides(max_consecutive_same_creator=3)
    posts = [post("1", "A", ["t1"]), post("2", "A", ["t2"]), post("3", "A", ["t3"]), post("4", "B", ["t4"])]
    assert ids(apply_diversity(posts, cfg)) == ["1", "2", "3", "4"]


def test_get_settings_with_overrides_is_usable() -> None:
    cfg = get_settings().with_overrides(max_consecutive_same_topic=1, max_consecutive_same_creator=5)
    posts = [post("1", "A", ["x"]), post("2", "B", ["x"]), post("3", "C", ["y"])]
    assert ids(apply_diversity(posts, cfg)) == ["1", "3", "2"]
