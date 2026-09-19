import numpy as np

from app.models import LocalHashEmbeddings


def test_local_hash_embeddings_are_stable_and_normalized() -> None:
    model = LocalHashEmbeddings(dim=128)
    first = np.asarray(model.embed_query("年假申请流程"))
    second = np.asarray(model.embed_query("年假申请流程"))

    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_local_hash_embeddings_distinguish_unrelated_text() -> None:
    model = LocalHashEmbeddings(dim=128)
    leave = np.asarray(model.embed_query("年假申请流程"))
    vpn = np.asarray(model.embed_query("VPN 登录错误"))

    assert float(np.dot(leave, vpn)) < 0.95
