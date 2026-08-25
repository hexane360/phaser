
import numpy

from phaser.utils.misc import create_compact_groupings, create_sparse_groupings


def test_create_sparse_groupings():
    assert [arr.tolist() for arr in create_sparse_groupings(32, 8, 'seed1')] == [
        [[23, 31,  0, 30, 29,  3,  4, 15]],
        [[14,  6,  9, 28, 13, 18, 17, 26]],
        [[24, 19, 10, 27,  5,  8, 20, 22]],
        [[ 1, 12,  2, 11,  7, 21, 25, 16]]
    ]

    # distributes evenly
    assert [arr.tolist() for arr in create_sparse_groupings(33, 8, 'seed1')] == [
        [[24, 16,  0, 30, 25,  3,  4]],
        [[15, 14,  6,  9,  7, 13, 21]],
        [[19, 31, 28, 17, 23, 18, 10]],
        [[29, 27, 32,  5,  8, 20]],
        [[22,  1, 26, 12,  2, 11]]
    ]

    # multi-dimensional
    arr = numpy.zeros((32, 32, 4))

    groupings = create_sparse_groupings(arr.shape, 128)
    assert groupings[0].shape == (3, 128)
    # make sure we can index
    assert arr[tuple(groupings[0])].shape == (128,)


def test_create_compact_groupings():
    scan = numpy.stack(numpy.meshgrid(
        numpy.linspace(0., 20., 8),
        numpy.linspace(0., 20., 8),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

    assert [arr.tolist() for arr in create_compact_groupings(scan, 16, 'seed1')] == [
        [[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]],
        [[32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59]],
        [[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]],
        [[36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63]],
    ]