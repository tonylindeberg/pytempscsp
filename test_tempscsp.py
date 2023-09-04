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


def test_explicitcascade():
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


def test_limitkern_composedsospars_alllayers():
    "Test limitkern_composedsospars_alllayers."
    muvec = np.array([
        6.103143141433787e-05,
        0.00018307195340983018,
        0.0007318862175645924,
        0.0029211543572212895,
        0.011584548242028148,
        0.04486236794258425,
        0.16143782776614768,
        0.5
    ])
    sospars_expected = np.array([[
        9.99755945e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.00000000e+00,
        -2.44066151e-04,
        1.11704165e-08
    ],[
        9.96358133e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.00000000e+00,
        -3.64399702e-03,
        2.13016647e-06
    ],[
        9.46103666e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.00000000e+00,
        -5.43880339e-02,
        4.91699788e-04
    ],[
        5.74001165e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.00000000e+00,
        -4.72331585e-01,
        4.63327506e-02
    ]])
    assert np.allclose(
        sospars_expected,
        tempscsp.limitkern_composedsospars_alllayers(muvec)
    )

def test_limitkern_composedsospars_alllayers_list():
    """Test tlimitkern_composedsospars_alllayers_list()."""
    muvec = np.array(
        [6.103143141433787e-05,
         0.00018307195340983018,
         0.0007318862175645924,
         0.0029211543572212895,
         0.011584548242028148,
         0.04486236794258425,
         0.16143782776614768,
         0.5
    ])
    listformat_expected = [0.9997559450194062, 0.0, 0.0, 1.0, -0.00024406615101033618, 1.1170416507132956e-08, 0.9963581331461244,
                 0.0, 0.0, 1.0, -0.003643997020350049, 2.1301664746710137e-06, 0.9461036658824216, 0.0, 0.0, 1.0,
                 -0.054388033905137426, 0.0004916997875589257, 0.5740011653907472, 0.0, 0.0, 1.0, -0.4723315852472125,
                 0.04633275063795974]
    assert np.allclose(
        listformat_expected,
        tempscsp.limitkern_composedsospars_alllayers_list(muvec)
    )


def test_limitkern_sospars_2layers():
    """test limitkern_sospars_2layers()."""
    res_expected = np.array([
        0.5740011653907472,
        0.0,
        0.0,
        1.0,
        -0.4723315852472125,
        0.04633275063795974
    ])
    assert np.allclose(
        res_expected,
        tempscsp.limitkern_sospars_2layers(mu1=0.16143782776614768, mu2=0.5)
    )

if __name__ == "__main__":
    test_limitkern_sospars_2layers()

    
