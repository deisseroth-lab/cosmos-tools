import pytest
import cosmos.imaging.atlas_registration as reg
import numpy as np

class Test_load_atlas:

    @classmethod
    def setup_class(self):
        atlas, annotations, atlas_outline = reg.load_atlas()
        self.atlas = atlas
        self.annotations = annotations
        self.atlas_outline = atlas_outline
        self.parent_atlas = reg.get_parent_atlas(self.atlas,
                                                 self.annotations,
                                                 use_grandparent=False)
        self.grandparent_atlas = reg.get_parent_atlas(self.atlas,
                                                 self.annotations,
                                                 use_grandparent=True)

    @classmethod
    def teardown_class(self):
        pass

    def test_1(self):
        assert self.atlas.shape == (456, 528)

    def test_2(self):
        assert self.atlas_outline.shape == (456, 528)

    def test_3(self):
        assert len(self.annotations) == 1327

    def test_4(self):
        assert self.annotations[str(500)]['acronym'].decode('utf-8') == 'MO'

    def test_5(self):
        assert self.annotations[str(254)]['acronym'].decode('utf-8') == 'RSP'

    def test_6(self):
        assert len(np.unique(self.parent_atlas)) == 51

    def test_7(self):
        assert len(np.unique(self.grandparent_atlas)) == 24

    def test_8(self):
        assert self.parent_atlas.shape == (456, 528)

    def test_9(self):
        assert self.grandparent_atlas.shape == (456, 528)


@pytest.mark.parametrize('img_coords, atlas_coords',
                         [(np.array([[26, 297],[430, 314]]),
                           np.array([[83, 227], [348, 227]]))])
def test_fit_atlas_transform(img_coords, atlas_coords):
    tform = reg.fit_atlas_transform(img_coords, atlas_coords)
    assert tform.scale == 1.5258774153884227
    assert tform.rotation == 0.04205439828667381
    assert tform.translation[0] == -85.9735849056604

# def test_align_atlas_to_image(atlas, img, atlas_coords, img_coords,
#                               do_debug=False):
#
#
# def test_assign_cells_to_regions(xy_coords, tform, atlas, annotations,
#                                  get_parent=False, do_debug=False)