import pandas as pd
import pytest

from book_trends.data import generate_sample, genre_monthly, load_and_aggregate


def test_sample_is_complete_monthly_series(tmp_path):
    path = tmp_path / "sample.csv"
    generated = generate_sample(path)
    monthly = load_and_aggregate(path)
    assert len(generated) == 360
    assert len(monthly) == 120
    assert pd.infer_freq(monthly.month) == "MS"
    assert monthly.titles_published.gt(0).all()


def test_title_level_input_counts_unique_titles_and_fills_gaps(tmp_path):
    path = tmp_path / "books.csv"
    pd.DataFrame({
        "publication_date": ["2020-01-05", "2020-01-20", "2020-01-20", "2020-03-02"],
        "title_id": ["a", "b", "b", "c"],
        "genre": ["Fiction", "Fiction", "Fiction", "Children"],
    }).to_csv(path, index=False)
    monthly = load_and_aggregate(path)
    assert monthly.titles_published.tolist() == [2.0, 0.0, 1.0]
    genres = genre_monthly(path)
    assert set(genres.genre) == {"Fiction", "Children"}


def test_invalid_dates_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"publication_date": ["not a date"], "title_id": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid"):
        load_and_aggregate(path)


def test_negative_counts_are_rejected(tmp_path):
    path = tmp_path / "neg.csv"
    pd.DataFrame({"publication_date": ["2020-01-01"], "titles_published": [-1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="negative"):
        load_and_aggregate(path)
