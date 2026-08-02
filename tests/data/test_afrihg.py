from sallm.data.afrihg import _load_csv_entries


def test_load_csv_entries_preserves_source_languages(tmp_path):
    xho = tmp_path / "xho.csv"
    zul = tmp_path / "zul.csv"
    xho.write_text("text,title\nxhosa article,xhosa title\n")
    zul.write_text("text,title\nzulu article,zulu title\n")

    dataset = _load_csv_entries([("xho", str(xho)), ("zul", str(zul))])

    assert dataset["lang"] == ["xho", "zul"]
