"""HFProposer mechanics on a tiny offline model (no downloads)."""

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import tiny_lm  # noqa: E402
from car2brick.decode import ConstrainedDecoder, DecodeConfig, HFProposer  # noqa: E402

TEXT = "<official>\ncar 20x8x12\n6 6/1 6/1\na plate 2x8 | red\n@root 178 120 7 r17\nb plate 2x10 | red\n" \
       "a stud stud m hole f 270\n@ 10 3 1\n"


@pytest.fixture(scope="module")
def proposer():
    tok = tiny_lm.build_tokenizer([TEXT] * 20, vocab_size=300)
    model = tiny_lm.build_model(len(tok))
    return HFProposer(model, tok, max_new_tokens=24)


def test_proposals_have_requested_lines(proposer):
    out = proposer("a plate 2x8 | red\n", 4, 2)
    assert len(out) == 4
    for o in out:
        assert o is None or len(o.split("\n")) <= 2


def test_loglik_is_finite(proposer):
    v = proposer.loglik("a plate", " 2x8 | red\n")
    assert math.isfinite(v) and v < 0


def test_decoder_runs_with_untrained_model(proposer, box_desc):
    res = ConstrainedDecoder(proposer, config=DecodeConfig(candidates=2, patience=1, max_parts=3)).run(box_desc)
    assert res.stats["stop"] in ("no_root", "stuck", "eos", "max_parts")
