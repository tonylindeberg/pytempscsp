"""Unit tests for tempscsp."""
import numpy as np
import tempscsp


def test_deltafcn1D():
    """Test the function returns expected output."""
    delta_vec = tempscsp.deltafcn1D(4)
    np.testing.assert_array_equal(
        delta_vec,
        np.array([1., 0., 0., 0.])
    )


def test_explictcascade():
    """Check explicit cascade for an example."""
    deltafcn = tempscsp.deltafcn1D(5)
    smooth1sos = tempscsp.limitkernfilt(deltafcn, 1, method='explicitcascade')
    assert np.allclose(
        smooth1sos,
        np.array([0.54095478, 0.28703476, 0.11196678, 0.03965, 0.01354296])
    )

def test_sosfilt():
    """Check sosfilter for an example."""
    deltafcn = tempscsp.deltafcn1D(5)
    smooth1sos = tempscsp.limitkernfilt(deltafcn, 1, method='sosfilt')
    print(smooth1sos)
    assert np.allclose(
        smooth1sos,
        np.array([0.54095478, 0.28703476, 0.11196678, 0.03965, 0.01354296])
    )


def test_mufromstddev():
    """Test mufromstddev."""
    muvec_expected = [6.103143141433787e-05,
         0.00018307195340983018,
         0.0007318862175645924,
         0.0029211543572212895,
         0.011584548242028148,
         0.04486236794258425,
         0.16143782776614768,
         0.5
    ]
    sigmavec_expected = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    muvec, sigmavec = tempscsp.mufromstddev(1, 2, 8)
    assert np.allclose(
        muvec_expected,
        muvec
    )
    assert np.allclose(
        sigmavec_expected,
        sigmavec
    )



if __name__ == "__main__":
    test_sosfilt()
